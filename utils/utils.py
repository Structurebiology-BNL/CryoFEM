import numpy as np
import logging
import sys
import torch
from models.data_load import CryoEM_Map_Dataset, CryoEM_Map_TestDataset


def load_data(conf, training=True):
    if training:
        id_path = conf["data"]["emd_id_path"]
    else:
        id_path = conf["test_data"]["emd_id_path"]
    with open(id_path) as file:
        lines = file.readlines()
        id_list = [line.rstrip() for line in lines]

    if training:
        RANDOM_SEED = int(conf["general"]["seed"])
        np.random.seed(RANDOM_SEED)
        val_size = int(conf["training"]["val_ratio"] * len(id_list))
        rng = np.random.default_rng(RANDOM_SEED)
        val_id = list(rng.choice(id_list, size=val_size, replace=False))
        train_id = list(set(id_list).difference(val_id))
        train_data = CryoEM_Map_Dataset(
            conf["data"]["data_path"],
            train_id,
            box_size=conf["data"]["box_size"],
            core_size=conf["data"]["core_size"],
            augmentation=conf["data"]["augmentation"],
            training=True,
        )
        val_data = CryoEM_Map_Dataset(
            conf["data"]["data_path"],
            val_id,
            box_size=conf["data"]["box_size"],
            core_size=conf["data"]["core_size"],
            augmentation=False,
            training=False,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=1, num_workers=16, pin_memory=True, shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=1, num_workers=16, pin_memory=True, shuffle=False
        )
        return train_dataloader, val_dataloader

    else:
        test_data = CryoEM_Map_TestDataset(
            conf["test_data"]["data_path"],
            id_list,
            box_size=conf["data"]["box_size"],
            core_size=conf["data"]["core_size"],
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=1, num_workers=16, pin_memory=True, shuffle=False
        )

        return test_dataloader


def logging_related(output_path=None, debug=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if not debug:
        assert output_path is not None, "need valid log output path"
        log_filename = str(output_path) + "/training.log"
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

    return np.mean((mean_xy - mean_x * mean_y) / (np.sqrt(var_x * var_y)) + 1e-10)
