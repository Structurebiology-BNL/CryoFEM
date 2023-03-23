import numpy as np
import logging
import sys
import torch
from torchvision.datasets.utils import download_url
import gemmi
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


def logging_related(output_path=None, debug=True, training=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if training:
        log_filename = str(output_path) + "/training.log"
    else:
        log_filename = str(output_path) + "/inference.log"
    if not debug:
        assert output_path is not None, "need valid log output path"

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


def download_half_maps(emdb_id):
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
        # /content/ResEM/ResEM/emd_23274_half_1.map.gz
        m1 = gemmi.read_ccp4_map("./emd_{}_half_1.map.gz".format(emdb_id))
        m1_arr = np.array(m1.grid, copy=False)
        m2 = gemmi.read_ccp4_map("./emd_{}_half_2.map.gz".format(emdb_id))
        m2_arr = np.array(m2.grid, copy=False)
        ## create a new map using gemmi
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = gemmi.FloatGrid((m1_arr + m2_arr) / 2)
        ccp4.grid.unit_cell.set(
            m1.grid.unit_cell.a,
            m1.grid.unit_cell.b,
            m1.grid.unit_cell.c,
            m1.grid.unit_cell.alpha,
            m1.grid.unit_cell.beta,
            m1.grid.unit_cell.gamma,
        )

        ccp4.grid.spacegroup = m1.grid.spacegroup
        ccp4.update_ccp4_header()
        ccp4.write_ccp4_map("averaged_map_{}.ccp4".format(emdb_id))
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