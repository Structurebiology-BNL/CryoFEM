import torch
import pytorch_lightning as pl
from .loss_func import Composite_Loss
from .unet import UNetRes


class UNetResLightning(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.model = UNetRes(n_blocks=conf.model.n_blocks, act_mode=conf.model.act_mode)
        self.conf = conf
        self.criterion = Composite_Loss(
            loss_1_type=conf.training.loss_1_type,
            beta=conf.training.smooth_l1_beta,
            cc_type=conf.training.cc_type,
            cc_weight=conf.training.cc_weight,
            device=self.device
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        indices, x_train, y_train = batch
        x_train = x_train.squeeze()
        y_train = y_train.squeeze()
        y_pred = torch.tensor(())
        for indx in range(0, x_train.shape[0], self.conf.training.batch_size):
            if indx + self.conf.training.batch_size > x_train.shape[0]:
                x_train_partial = x_train[indx:].unsqueeze(dim=1)
                y_train_partial = y_train[indx:].unsqueeze(dim=1)
            else:
                x_train_partial = x_train[indx : indx + self.conf.training.batch_size].unsqueeze(dim=1)
                y_train_partial = y_train[indx : indx + self.conf.training.batch_size].unsqueeze(dim=1)
            y_pred_partial = self.model(x_train_partial)
            loss = self.criterion(y_pred_partial, y_train_partial)
            y_pred = torch.cat((y_pred, y_pred_partial.squeeze(dim=1).detach()), dim=0)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        indices, x_val, y_val = batch
        x_val = x_val.squeeze()
        y_val = y_val.squeeze()
        y_val_pred = torch.tensor(())
        for indx in range(0, x_val.shape[0], self.conf.training.batch_size):
            if indx + self.conf.training.batch_size > x_val.shape[0]:
                x_val_partial = x_val[indx:].unsqueeze(dim=1)
            else:
                x_val_partial = x_val[indx : indx + self.conf.training.batch_size].unsqueeze(dim=1)
            x_val_partial = x_val[indx : indx + self.conf.training.batch_size].unsqueeze(dim=1)
            y_pred_partial = self.model(x_val_partial)
            y_val_pred = torch.cat((y_val_pred, y_pred_partial.squeeze(dim=1).detach()), dim=0)

        val_loss = self.criterion(y_val_pred, y_val)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training.lr,
            weight_decay=self.conf.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.conf.training.scheduler_step_size,
            gamma=self.conf.training.lr_decay,
        )
        return [optimizer], [scheduler]
