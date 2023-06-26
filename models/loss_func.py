import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Composite_Loss(nn.Module):
    def __init__(
        self,
        loss_1_type="l1",
        cc_weight=0.5,
        cc_type="pcc",
        beta=0.05,
        kernel_size=7,
        device="cpu",
    ):
        super(Composite_Loss, self).__init__()
        assert loss_1_type in ["l1", "l2", "smooth_l1"]
        assert cc_type in ["pcc", "mask_pcc", "kernel_pcc"]
        self.cc_weight = cc_weight
        self.beta = beta
        self.cc_type = cc_type
        if loss_1_type == "l1":
            self.loss_1 = nn.L1Loss()
        elif loss_1_type == "l2":
            self.loss_1 = nn.MSELoss()
        else:
            self.loss_1 = torch.nn.SmoothL1Loss(beta=self.beta)
        if self.cc_type == "kernel_pcc":
            self.kernel = create_soft_edged_kernel_pxl(kernel_size)
            self.kernel = self.kernel.to(device)

    def forward(self, input, target):
        loss_1 = self.loss_1(input, target)
        if self.cc_type == "mask_pcc":
            cc_ = masked_pcc_loss(input, target)
        elif self.cc_type == "kernel_pcc":
            cc_ = kernel_pcc_loss(input, target, self.kernel)
        elif self.cc_type == "pcc":
            cc_ = pcc_loss(input, target)
        else:
            raise ValueError("Unknown cc_type: {}".format(self.cc_type))
        loss = (loss_1 + self.cc_weight * cc_) / (1 + self.cc_weight)

        return loss_1, cc_, loss


def create_soft_edged_kernel_pxl(r1):
    # Create soft-edged-kernel. r1 is the radius of kernel in pixels
    from math import sqrt, cos

    if r1 < 3:
        boxsize = 5
    else:
        boxsize = 2 * r1 + 1
    kern_sphere_soft = np.zeros(shape=(boxsize, boxsize, boxsize), dtype="float")
    kx = ky = kz = boxsize
    center = boxsize // 2
    # print("center: ", center)
    r1 = center
    r0 = r1 - 2
    # print("r1: ", r1, "r0: ", r0)
    for i in range(kx):
        for j in range(ky):
            for k in range(kz):
                dist = sqrt((i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2)
                if dist < r1:
                    if dist < r0:
                        kern_sphere_soft[i, j, k] = 1
                    else:
                        kern_sphere_soft[i, j, k] = (
                            (1 + cos(np.pi * (dist - r0) / (r1 - r0)))
                        ) / 2.0
    kern_sphere_soft = kern_sphere_soft / np.sum(kern_sphere_soft)

    return torch.Tensor(kern_sphere_soft).unsqueeze(dim=0).unsqueeze(dim=0)


def kernel_pcc_loss(output, target, kernel):
    if target.dim() == 3:  # when calculating the kernel_pcc for the original map
        target = target.unsqueeze(dim=0).unsqueeze(dim=0).clone()
        output = output.unsqueeze(dim=0).unsqueeze(dim=0).clone()
    if target.sum() == 0.0:  # pcc is not defined
        return 0.5
    else:
        num_element = torch.numel(output[0, 0, :, :, :])
        # determine if the block is dense (non-zero elements > 0.01%)
        sparsity = (torch.count_nonzero(target, dim=(2, 3, 4)) / num_element) > 0.0001
        loc3_A = F.conv3d(output, kernel, padding="same")
        loc3_A2 = F.conv3d(output * output, kernel, padding="same")
        loc3_B = F.conv3d(target, kernel, padding="same")
        loc3_B2 = F.conv3d(target * target, kernel, padding="same")
        loc3_AB = F.conv3d(output * target, kernel, padding="same")
        cov3_AB = loc3_AB - loc3_A * loc3_B
        var3_A = loc3_A2 - loc3_A**2
        var3_B = loc3_B2 - loc3_B**2
        reg_a = torch.max(var3_A) / 1000
        reg_b = torch.max(var3_B) / 1000
        var3_A = torch.where(var3_A < reg_a, reg_a, var3_A)
        var3_B = torch.where(var3_B < reg_b, reg_b, var3_B)
        kernel_weighted_cc = cov3_AB / torch.sqrt(var3_A * var3_B)
        kernel_weighted_cc = torch.mean(kernel_weighted_cc, dim=(2, 3, 4))
        # kernel_weighted_cc_loss = torch.mean(sparsity * (1 - kernel_weighted_cc))

        # for sparse blocks, returns default loss 0.5, for dense blocks, return 1 - pcc
        kernel_weighted_cc_loss = torch.mean(
            0.5 * (1.0 - sparsity.float()) + sparsity * (1 - kernel_weighted_cc)
        )

        return kernel_weighted_cc_loss


def masked_pearson_corr(x, y):
    if y.sum() == 0:  # pcc is not defined
        return 0.5
    mask = y != 0
    x_masked = x[mask]
    y_masked = y[mask]

    x_mean = x_masked.mean()
    y_mean = y_masked.mean()

    x_centered = x_masked - x_mean
    y_centered = y_masked - y_mean

    covariance = (x_centered * y_centered).sum()
    x_var = (x_centered**2).sum()
    y_var = (y_centered**2).sum()

    correlation = covariance / torch.sqrt(x_var * y_var + 1e-12)
    return correlation


def masked_pcc_loss(x, y):
    x = x.squeeze()
    y = y.squeeze()
    correlations = torch.tensor(
        [masked_pearson_corr(x[i], y[i]) for i in range(x.shape[0])]
    )

    # Calculate the average correlation
    avg_correlation = torch.mean(correlations)
    return 1 - avg_correlation


def pcc_loss(output, target):
    ## Pearson correlation coefficient
    if output.dim() > 3:
        x, y = output.squeeze(dim=1), target.squeeze(dim=1)
    else:
        x, y = output, target
    if y.sum() == 0:  # pcc is not defined
        return 0.5
    else:
        mean_x = torch.mean(x, dim=(-1, -2, -3))
        mean_y = torch.mean(y, dim=(-1, -2, -3))
        var_x = (
            torch.mean(x**2, dim=(-1, -2, -3))
            - (torch.mean(x, dim=(-1, -2, -3))) ** 2
        )
        var_y = (
            torch.mean(y**2, dim=(-1, -2, -3))
            - (torch.mean(y, dim=(-1, -2, -3))) ** 2
        )
        mean_xy = torch.mean(x * y, dim=(-1, -2, -3))
        pcc = torch.mean(
            (mean_xy - mean_x * mean_y) / torch.sqrt(var_x * var_y + 1e-12)
        )

        return 1 - pcc
