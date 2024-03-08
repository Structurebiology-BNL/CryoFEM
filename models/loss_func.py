import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Composite_Loss(nn.Module):
    def __init__(
        self,
        loss_1_type="l1",
        cc_weight=0.1,
        beta=0.05,
    ):
        super(Composite_Loss, self).__init__()
        assert loss_1_type in ["l1", "l2", "smooth_l1"]
        self.cc_weight = cc_weight
        self.beta = beta
        if loss_1_type == "l1":
            self.loss_1 = nn.L1Loss()
        elif loss_1_type == "l2":
            self.loss_1 = nn.MSELoss()
        else:
            self.loss_1 = torch.nn.SmoothL1Loss(beta=self.beta)

    def forward(self, input, target):
        loss_1 = self.loss_1(input, target)
        cc_ = pcc_loss(input, target)
        loss = (loss_1 + self.cc_weight * cc_) / (1 + self.cc_weight)

        return loss_1, cc_, loss


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
            torch.mean(x**2, dim=(-1, -2, -3)) - (torch.mean(x, dim=(-1, -2, -3))) ** 2
        )
        var_y = (
            torch.mean(y**2, dim=(-1, -2, -3)) - (torch.mean(y, dim=(-1, -2, -3))) ** 2
        )
        mean_xy = torch.mean(x * y, dim=(-1, -2, -3))
        pcc = torch.mean(
            (mean_xy - mean_x * mean_y) / torch.sqrt(var_x * var_y + 1e-12)
        )

        return 1 - pcc
