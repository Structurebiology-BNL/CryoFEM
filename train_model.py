import torch
import argparse
import json
import os
import logging
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from tqdm import tqdm
from models.map_splitter import reconstruct_maps
from models.unet import UNetRes
from utils.utils import (
    load_data,
    EarlyStopper,
    pearson_cc,
    logging_related,
    peak_signal_to_noise_ratio,
    process_config,
)
from models.loss_func import Composite_Loss


def train(conf):
    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    # load data
    train_dataloader, val_dataloader = load_data(conf, training=True)
    logging.info(
        "Total train samples {}, val samples {}".format(
            len(train_dataloader), len(val_dataloader)
        )
    )
    # load model, optimizer, scheduler
    model = UNetRes(n_blocks=conf.model.n_blocks, act_mode=conf.model.act_mode).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=conf.training.lr, weight_decay=conf.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=conf.training.scheduler_step_size,
        gamma=conf.training.lr_decay,
    )
    # if using PyTorch 2.0, use torch.compile to accelerate the training
    if float(torch.__version__[:3]) >= 2.0:
        logging.info("Using PyTorch 2.0, use torch.compile to accelerate the training")
        model = torch.compile(model)
    if conf.training.load_checkpoint:
        logging.info(
            "Resume training and load model from {}".format(
                conf.training.load_checkpoint
            )
        )
        checkpoint = torch.load(conf.training.load_checkpoint)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    # define loss function
    criterion = Composite_Loss(
        loss_1_type=conf.training.loss_1_type,
        beta=conf.training.smooth_l1_beta,
        cc_weight=conf.training.cc_weight,
    )
    batch_size = conf.training.batch_size
    torch.backends.cudnn.benchmark = True
    early_stopper = EarlyStopper(patience=6, mode="min")
    best_models = []
    for epoch in range(1, conf.training.epochs + 1):
        model.train()
        train_loss, pcc, psnr = 0.0, 0.0, 0.0
        train_loss_l1, train_loss_cc = 0.0, 0.0
        for x_train, y_train, original_shape, id in tqdm(train_dataloader):
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
                _, _, loss_ = criterion(y_pred_partial, y_train_partial)
                loss_.backward()
                y_pred = torch.cat(
                    (y_pred, y_pred_partial.squeeze(dim=1).detach().cpu()), dim=0
                )
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
            tmp_pcc = pearson_cc(y_pred_recon, y_train_recon)
            tmp_psnr = peak_signal_to_noise_ratio(y_pred_recon, y_train_recon)
            tmp_loss_l1, tmp_loss_cc, tmp_loss = criterion(
                torch.from_numpy(y_pred_recon).to(device),
                torch.from_numpy(y_train_recon).to(device),
            )
            pcc += tmp_pcc
            psnr += tmp_psnr
            train_loss += tmp_loss.detach().cpu().numpy()
            train_loss_l1 += tmp_loss_l1.detach().cpu().numpy()
            train_loss_cc += tmp_loss_cc.detach().cpu().numpy()
            logging.info(
                "Epoch {}, running loss: {:.4f}, EMDB-{} psnr: {:.2f},\n"
                "pcc: {:.4f}, l1 loss: {:.4f}, cc loss: {:.4f}".format(
                    epoch,
                    tmp_loss,
                    id[0],
                    tmp_psnr,
                    tmp_pcc,
                    tmp_loss_l1,
                    tmp_loss_cc,
                )
            )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        train_loss = train_loss / len(train_dataloader)
        train_pcc = pcc / len(train_dataloader)
        train_psnr = psnr / len(train_dataloader)
        writer.add_scalars(
            "train",
            {
                "loss": train_loss,
                "pcc": train_pcc,
                "psnr": train_psnr,
                "lr": lr,
            },
            epoch,
        )
        """
        Validation
        """
        model.eval()
        with torch.no_grad():
            val_loss, pcc, psnr = 0.0, 0.0, 0.0
            val_loss_l1, val_loss_cc = 0.0, 0.0
            for x_val, y_val, original_shape, id in val_dataloader:
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
                tmp_pcc = pearson_cc(y_val_pred_recon, y_val_recon)
                tmp_psnr = peak_signal_to_noise_ratio(y_val_pred_recon, y_val_recon)
                tmp_loss_l1, tmp_loss_cc, tmp_loss = criterion(
                    torch.from_numpy(y_val_pred_recon).to(device),
                    torch.from_numpy(y_val_recon).to(device),
                )
                pcc += tmp_pcc
                psnr += tmp_psnr
                val_loss += tmp_loss.detach().cpu().numpy()
                val_loss_l1 += tmp_loss_l1.detach().cpu().numpy()
                val_loss_cc += tmp_loss_cc.detach().cpu().numpy()
                logging.info(
                    "Epoch {}, running validation loss: {:.4f}, EMDB-{} psnr: {:.2f},\n"
                    "pcc: {:.4f}, l1 loss: {:.4f}, cc loss: {:.4f}".format(
                        epoch,
                        tmp_loss,
                        id[0],
                        tmp_psnr,
                        tmp_pcc,
                        tmp_loss_l1,
                        tmp_loss_cc,
                    )
                )
            val_loss = val_loss / len(val_dataloader)
            val_psnr = psnr / len(val_dataloader)
            val_pcc = pcc / len(val_dataloader)
            writer.add_scalars(
                "val",
                {
                    "loss": val_loss,
                    "pcc": val_pcc,
                    "psnr": val_psnr,
                },
                epoch,
            )
            if early_stopper.early_stop(val_loss):
                logging.info("Early stopping at epoch {}...".format(epoch))
                break
            if (
                len(best_models) < 2
                or val_loss < best_models[-1][0]
                and not conf.general.debug
            ):  # save the best top-2 models
                state = {
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                file_name = (
                    conf.output_path
                    + "/"
                    + "Epoch_{}".format(epoch)
                    + "_psnr_{:.2f}".format(val_psnr)
                    + "_pcc_{:.3f}".format(val_pcc)
                )
                logging.info("\n------------ Save the best model ------------")
                torch.save(state, file_name + ".pt")

                # Remove the lowest scoring model if we already have 2 models
                if len(best_models) == 2:
                    _, old_file_name = best_models.pop()
                    os.remove(old_file_name + ".pt")

                # Add the new model to the list and sort it
                best_models.append((val_loss, file_name))
                best_models.sort(key=lambda x: x[0])

        logging.info(
            "Epoch {} train loss: {:.4f}, val loss: {:.4f}, lr = {:.1e}\n"
            "train psnr: {:.2f}, val psnr {:.2f}\n"
            "train pcc: {:.4f}, val pcc: {:.4f}\n".format(
                epoch,
                train_loss,
                val_loss,
                lr,
                train_psnr,
                val_psnr,
                train_pcc,
                val_pcc,
            )
        )


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Name of training configuration file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        conf = json.load(f)
    conf = process_config(conf, config_name=Path(args.config).stem)
    """
    logging related part
    """
    logging_related(rank=0, output_path=conf.output_path)
    writer = SummaryWriter(log_dir=conf.output_path)
    train(conf)
    writer.flush()
    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
