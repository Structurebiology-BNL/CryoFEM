import torch
import pytorch_lightning as pl
from .loss_func import Composite_Loss
from .unet import UNetRes

class UNetResLightning(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.model = UNetRes(n_blocks=conf.model.n_blocks, act_mode=conf.model.act_mode)
        self.criterion = Composite_Loss(
            loss_1_type=conf.training.loss_1_type,
            beta=conf.training.smooth_l1_beta,
            cc_type=conf.training.cc_type,
            cc_weight=conf.training.cc_weight,
            device=self.device,
        )
        self.save_hyperparameters(conf)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_train, y_train, original_shape, _ = batch
        y_pred = self(x_train)
        loss = self.criterion(y_pred, y_train)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_val, y_val, original_shape, _ = batch
        y_pred = self(x_val)
        loss = self.criterion(y_pred, y_val)
        self.log("val_loss", loss)
        return loss

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