import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import argparse
import json
import logging
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from skimage.metrics import structural_similarity
from tqdm import tqdm
from models.map_splitter import reconstruct_maps
from models.unet import UNetRes, UNet
from utils.utils import (
    load_data_ddp,
    EarlyStopper,
    pearson_cc,
    logging_related,
    peak_signal_to_noise_ratio,
    process_config,
)
from models.loss_func import Composite_Loss


# Other imports and definitions for your dataset, model, criterion, and metrics


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group(
        "nccl", init_method="env://", rank=rank, world_size=world_size
    )


def cleanup():
    dist.destroy_process_group()


def train_ddp(rank, conf):
    setup(rank, conf.general.num_gpus)
    """
    logging related part
    """
    logging_related(rank, output_path=conf.output_path)
    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:{}".format(rank))
    train_dataloader, val_dataloader = load_data_ddp(
        conf, rank, world_size=conf.general.num_gpus, training=True
    )
    if conf.model.model_type == "unetres":
        model = UNetRes(n_blocks=conf.model.n_blocks, act_mode=conf.model.act_mode).to(
            device
        )

    elif conf.model.model_type == "unet":
        model = UNet(n_blocks=conf.model.n_blocks, act_mode=conf.model.act_mode).to(
            device
        )
    model.to(device)
    model = DDP(model, device_ids=[device])
    lr = conf.general.num_gpus * conf.training.lr
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=conf.training.weight_decay
    )
    step_size = conf.training.scheduler_step_size
    lr_decay = conf.training.lr_decay
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=lr_decay
    )
    writer = SummaryWriter(log_dir=conf.output_path) if rank == 0 else None
    if conf.model.load_checkpoint:
        logging.info(
            "Resume training and load model from {}".format(conf.model.load_checkpoint)
        )
        checkpoint = torch.load(conf.model.load_checkpoint)
        model.load_state_dict(checkpoint.model_state)
        optimizer.load_state_dict(checkpoint.optimizer)
        scheduler.load_state_dict(checkpoint.scheduler)

    logging.info(
        "Total train samples {}, val samples {} for rank {}".format(
            len(train_dataloader), len(val_dataloader), rank
        )
    )
    criterion = Composite_Loss(
        loss_1_type=conf.training.loss_1_type,
        beta=conf.training.smooth_l1_beta,
        cc_type=conf.training.cc_type,
        cc_weight=conf.training.cc_weight,
        device=device,
    )
    EPOCHS = conf.training.epochs
    batch_size = conf.training.batch_size
    torch.backends.cudnn.benchmark = True
    early_stopper = EarlyStopper(patience=6, mode="min")
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss, struc_sim, pcc, psnr = 0.0, 0.0, 0.0, 0.0
        for i, (x_train, y_train, original_shape, id) in enumerate(
            tqdm(train_dataloader)
        ):
            x_train = x_train.squeeze()
            y_train = y_train.squeeze()
            y_pred = torch.tensor(())
            for indx in range(0, x_train.shape[0], batch_size):
                if indx + batch_size > x_train.shape[0]:
                    x_train_partial = x_train[indx:].unsqueeze(dim=1).to(device)
                    y_train_partial = y_train[indx:].unsqueeze(dim=1).to(device)
                else:
                    x_train_partial = (
                        x_train[indx : indx + batch_size].unsqueeze(dim=1).to(device)
                    )
                    y_train_partial = (
                        y_train[indx : indx + batch_size].unsqueeze(dim=1).to(device)
                    )
                optimizer.zero_grad(set_to_none=True)
                y_pred_partial = model(x_train_partial)
                loss_ = criterion(y_pred_partial, y_train_partial)
                y_pred = torch.cat(
                    (y_pred, y_pred_partial.squeeze(dim=1).detach().cpu()), dim=0
                )
                loss_.backward()
                clip_grad_norm_(model.parameters(), 2)
                optimizer.step()

            y_pred_recon = reconstruct_maps(
                y_pred.numpy(),
                original_shape,
                box_size=conf.data.box_size,
                core_size=conf.data.core_size,
            )
            y_train_recon = reconstruct_maps(
                y_train.detach().cpu().numpy(),
                original_shape,
                box_size=conf.data.box_size,
                core_size=conf.data.core_size,
            )

            struc_sim += structural_similarity(y_pred_recon, y_train_recon)
            pcc += pearson_cc(y_pred_recon, y_train_recon)
            psnr += peak_signal_to_noise_ratio(y_pred_recon, y_train_recon)
            train_loss += (
                criterion(
                    torch.from_numpy(y_pred_recon).to(device),
                    torch.from_numpy(y_train_recon).to(device),
                )
                .detach()
                .cpu()
                .numpy()
            )

            logging.info(
                "Rank {}, Epoch {}, running loss: {:.4f}, EMDB-{} ssim: {:.4f},\n"
                "psnr: {:.2f}, pcc: {:.4f}".format(
                    rank,
                    epoch + 1,
                    train_loss / (i + 1),
                    id[0],
                    struc_sim / (i + 1),
                    psnr / (i + 1),
                    pcc / (i + 1),
                )
            )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        train_loss = train_loss / len(train_dataloader)
        train_struc_sim = struc_sim / len(train_dataloader)
        train_pcc = pcc / len(train_dataloader)
        train_psnr = psnr / len(train_dataloader)
        if writer is not None:
            writer.add_scalars(
                "train",
                {
                    "loss": train_loss,
                    "struc_sim": train_struc_sim,
                    "pcc": train_pcc,
                    "psnr": train_psnr,
                    "lr": lr,
                },
                epoch + 1,
            )
        """
        Validation
        """
        model.eval()
        with torch.no_grad():
            val_loss, struc_sim, pcc, psnr = 0.0, 0.0, 0.0, 0.0
            for i, (x_val, y_val, original_shape, id) in enumerate(val_dataloader):
                x_val = x_val.squeeze()
                y_val = y_val.squeeze()
                y_val_pred = torch.tensor(())
                for indx in range(0, x_val.shape[0], batch_size):
                    if indx + batch_size > x_val.shape[0]:
                        x_val_partial = x_val[indx:].unsqueeze(dim=1).to(device)
                    else:
                        x_val_partial = (
                            x_val[indx : indx + batch_size].unsqueeze(dim=1).to(device)
                        )
                    y_pred_partial = model(x_val_partial)
                    y_val_pred = torch.cat(
                        (y_val_pred, y_pred_partial.squeeze(dim=1).detach().cpu()),
                        dim=0,
                    )

                y_val_pred_recon = reconstruct_maps(
                    y_val_pred.numpy(),
                    original_shape,
                    box_size=conf.data.box_size,
                    core_size=conf.data.core_size,
                )
                y_val_recon = reconstruct_maps(
                    y_val.numpy(),
                    original_shape,
                    box_size=conf.data.box_size,
                    core_size=conf.data.core_size,
                )
                struc_sim += structural_similarity(y_val_pred_recon, y_val_recon)
                pcc += pearson_cc(y_val_pred_recon, y_val_recon)
                psnr += peak_signal_to_noise_ratio(y_val_pred_recon, y_val_recon)
                val_loss += (
                    criterion(
                        torch.from_numpy(y_val_pred_recon).to(device),
                        torch.from_numpy(y_val_recon).to(device),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                logging.info(
                    "Epoch {}, running validation loss: {:.4f}, EMDB-{} ssim: {:.4f},\n"
                    "psnr: {:.2f}, pcc: {:.4f}".format(
                        epoch + 1,
                        val_loss / (i + 1),
                        id[0],
                        struc_sim / (i + 1),
                        psnr / (i + 1),
                        pcc / (i + 1),
                    )
                )
            val_loss = val_loss / len(val_dataloader)
            val_struc_sim = struc_sim / len(val_dataloader)
            val_psnr = psnr / len(val_dataloader)
            val_pcc = pcc / len(val_dataloader)
            if writer is not None:
                writer.add_scalars(
                    "val",
                    {
                        "loss": val_loss,
                        "struc_sim": val_struc_sim,
                        "pcc": val_pcc,
                        "psnr": val_psnr,
                    },
                    epoch + 1,
                )
            if early_stopper.early_stop(val_loss):
                break
            if val_loss < early_stopper.best_metric and not conf.general.debug:
                state = {
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                torch.save(state, "best_model.pt")
                logging.info("\n----Saved the best model----\n")

        logging.info(
            "Epoch {} train loss: {:.4f}, val loss: {:.4f},\n"
            "train ssim: {:.4f}, val ssim: {:.4f}, lr = {}\n"
            "train psnr: {:.2f}, val psnr {:.2f}\n"
            "train pcc: {:.4f}, val pcc: {:.4f}\n".format(
                epoch + 1,
                train_loss,
                val_loss,
                train_struc_sim,
                val_struc_sim,
                lr,
                train_psnr,
                val_psnr,
                train_pcc,
                val_pcc,
            )
        )

    cleanup()


def main(conf):
    mp.set_start_method("spawn")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = conf.general.num_gpus
    mp.spawn(train_ddp, args=(conf,), nprocs=world_size, join=True)


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Name of configuration file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        conf = json.load(f)
    conf = process_config(conf, config_name=Path(args.config).stem)
    main(conf)
    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
