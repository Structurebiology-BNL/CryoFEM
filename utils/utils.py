import numpy as np
import logging
import sys
import json
from pathlib import Path
import datetime
from ml_collections import config_dict
from torchvision.datasets.utils import download_url
from models.data_load import CryoEM_Map_Dataset, CryoEM_Map_TestDataset
from torch.utils.data import DataLoader, DistributedSampler


def load_data_ddp(conf, rank, world_size, training=True):
    with open(conf.data.emd_id_path) as file:
        lines = file.readlines()
        id_list = [line.rstrip() for line in lines]

    if training:
        RANDOM_SEED = int(conf.general.seed)
        np.random.seed(RANDOM_SEED)
        val_size = int(conf.training.val_ratio * len(id_list))
        rng = np.random.default_rng(RANDOM_SEED)
        val_id = list(rng.choice(id_list, size=val_size, replace=False))
        train_id = list(set(id_list).difference(val_id))
        train_data = CryoEM_Map_Dataset(
            conf.data.data_path,
            train_id,
            box_size=conf.data.box_size,
            core_size=conf.data.core_size,
            augmentation=conf.data.augmentation,
            training=True,
        )
        val_data = CryoEM_Map_Dataset(
            conf.data.data_path,
            val_id,
            box_size=conf.data.box_size,
            core_size=conf.data.core_size,
            augmentation=False,
            training=False,
        )
        train_sampler = DistributedSampler(
            train_data, num_replicas=world_size, rank=rank
        )
        train_dataloader = DataLoader(
            train_data,
            batch_size=1,
            sampler=train_sampler,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
        )
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(
            val_data,
            batch_size=1,
            sampler=val_sampler,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
        )
        return train_dataloader, val_dataloader

    else:
        test_data = CryoEM_Map_TestDataset(
            conf.test_data.data_path,
            id_list,
            box_size=conf.data.box_size,
            core_size=conf.data.core_size,
        )
        test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank)
        test_dataloader = DataLoader(
            test_data,
            batch_size=1,
            sampler=test_sampler,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
        )

        return test_dataloader


def load_data(conf, training=True):
    if training:
        with open(conf.data.train_id_path) as file:
            lines = file.readlines()
            train_id = [line.rstrip() for line in lines]
        with open(conf.data.val_id_path) as file:
            lines = file.readlines()
            val_id = [line.rstrip() for line in lines]

        train_data = CryoEM_Map_Dataset(
            conf.data.data_path,
            train_id,
            box_size=conf.data.box_size,
            core_size=conf.data.core_size,
            augmentation=conf.data.augmentation,
            training=True,
        )
        val_data = CryoEM_Map_Dataset(
            conf.data.data_path,
            val_id,
            box_size=conf.data.box_size,
            core_size=conf.data.core_size,
            augmentation=False,
            training=False,
        )

        train_dataloader = DataLoader(
            train_data, batch_size=1, num_workers=16, pin_memory=True, shuffle=True
        )
        val_dataloader = DataLoader(
            val_data, batch_size=1, num_workers=16, pin_memory=True, shuffle=False
        )
        return train_dataloader, val_dataloader

    else:
        with open(conf.test_data.emd_id_path) as file:
            lines = file.readlines()
            id_list = [line.rstrip() for line in lines]
        test_data = CryoEM_Map_TestDataset(
            conf.test_data.data_path,
            id_list,
            box_size=conf.data.box_size,
            core_size=conf.data.core_size,
        )

        test_dataloader = DataLoader(
            test_data, batch_size=1, num_workers=16, pin_memory=True, shuffle=False
        )

        return test_dataloader

def process_config(conf, config_name="train", training=True):
    if training:
        if "load_checkpoint" in conf["training"]:
            model_config_path = (
                "/".join(conf["training"]["load_checkpoint"].split("/")[:-1])
                + "/config.json"
            )
        with open(model_config_path, "r") as f:
            conf_model = json.load(f)
        conf["model"] = conf_model["model"]

    output_path = None
    if not conf["general"]["debug"]:
        output_path = (
            Path("./results/")
            / config_name
            / Path(
                str(datetime.datetime.now())[:16].replace(" ", "-").replace(":", "-")
            )
        )
        output_path.mkdir(parents=True, exist_ok=True)
        conf["output_path"] = "./" + str(output_path)
        with open(str(output_path) + "/config.json", "w") as f:
            json.dump(conf, f, indent=4)

    conf = config_dict.ConfigDict(conf)
    return conf


def logging_related(rank, output_path=None, debug=False, training=True):
    logger = logging.getLogger()

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if not debug and rank == 0:
        assert output_path is not None, "need valid log output path"

        if training:
            log_filename = str(output_path) + "/training.log"
        else:
            log_filename = str(output_path) + "/inference.log"

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info("Output path: {}".format(output_path))


def peak_signal_to_noise_ratio(output, target):
    mse = np.mean((output - target) ** 2, axis=(0, 1, 2))
    return np.mean(20 * np.log10(1 / np.sqrt(mse)))


def pearson_cc(x, y):
    # Pearson correlation coefficient
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.mean(x**2) - (np.mean(x)) ** 2
    var_y = np.mean(y**2) - (np.mean(y)) ** 2
    mean_xy = np.mean(x * y)

    return np.mean((mean_xy - mean_x * mean_y) / np.sqrt(var_x * var_y + 1e-12))


def download_half_maps(emdb_id):
    import mrcfile

    half_map_1 = "https://files.wwpdb.org/pub/emdb/structures/EMD-{}/other/emd_{}_half_map_1.map.gz".format(
        emdb_id, emdb_id
    )
    half_map_2 = "https://files.wwpdb.org/pub/emdb/structures/EMD-{}/other/emd_{}_half_map_2.map.gz".format(
        emdb_id, emdb_id
    )
    download_successful = False
    try:
        download_url(half_map_1, "./", filename="emd_{}_half_1.map.gz".format(emdb_id))
        download_url(half_map_2, "./", filename="emd_{}_half_2.map.gz".format(emdb_id))
        m1 = mrcfile.open("emd_{}_half_map_1.map.gz".format(emdb_id), mode="r")
        meta_data = m1.header
        m2 = mrcfile.open("emd_{}_half_map_2.map.gz".format(emdb_id), mode="r")
        average = 0.5 * (m1.data + m2.data)
        ## create a new map using mrcfile
        with mrcfile.new("averaged_map_{}.mrc".format(emdb_id)) as mrc:
            mrc.set_data(average)
            mrc.header.cella.x = meta_data.cella.x
            mrc.header.cella.y = meta_data.cella.y
            mrc.header.cella.z = meta_data.cella.z
            mrc.header.nxstart = meta_data.nxstart
            mrc.header.nystart = meta_data.nystart
            mrc.header.nzstart = meta_data.nzstart

        download_successful = True
    except:
        print("no half maps available for the EMBD-{}".format(emdb_id))

    return download_successful


class EarlyStopper:
    def __init__(self, patience=6, min_delta=1e-7, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        if self.mode == "max":
            self.best_metric = -np.inf
        elif self.mode == "min":
            self.best_metric = 1e5

    def early_stop(self, validation_metric):
        if self.mode == "max":
            if validation_metric > self.best_metric:
                self.best_metric = validation_metric
                self.counter = 0
            elif validation_metric <= (self.best_metric - self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True

        elif self.mode == "min":
            if validation_metric < self.best_metric:
                self.best_metric = validation_metric
                self.counter = 0
            elif validation_metric >= (self.best_metric + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True

        return False
