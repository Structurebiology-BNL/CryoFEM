import pytorch_lightning as pl
import torch 
import argparse
import json
import logging
from pathlib import Path
from timeit import default_timer as timer
from utils.utils import (
    load_data,
    logging_related,
    process_config,
)
from models.unet_lightning import UNetResLightning


def train(conf):
    RANDOM_SEED = int(conf.general.seed)
    pl.seed_everything(RANDOM_SEED)
    train_dataloader, val_dataloader = load_data(conf, training=True)
    model = UNetResLightning(conf)
    trainer = pl.Trainer(
        gpus=[conf.general.gpu_id] if torch.cuda.is_available() else None,
        max_epochs=conf.training.epochs,
        precision=16,
        gradient_clip_val=2,
        auto_lr_find=True,
        auto_scale_batch_size="power",
        progress_bar_refresh_rate=20,
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            dirpath=conf.output_path,
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        early_stop_callback=pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, mode="min"
        ),
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    
if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Name of configuration file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        conf = json.load(f)
    conf, output_path = process_config(conf, config_name=Path(args.config).stem)
    """
    logging related part
    """
    logging_related(output_path=output_path, debug=conf.general.debug)
    train(conf)
    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
