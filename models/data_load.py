import torch
import numpy as np
import os
import mrcfile
import pathlib
from copy import deepcopy
import torchio
from .map_splitter import create_cube_list


class CryoEM_Map_Dataset(torch.utils.data.Dataset):
    """
    Load the original map (normalized) and simulated map as
    input and target, respectively
    """

    def __init__(
        self,
        data_dir,
        id_list,
        box_size=64,
        core_size=50,
        augmentation=False,
        training=False,
    ):
        self.data_dir = data_dir
        self.id_list = id_list
        self.box_size = box_size
        self.core_size = core_size
        assert (augmentation and training) == True or (
            (not augmentation and not training)
        ), "augmentation and training should be on or off together"
        self.augmentation = augmentation
        self.training = training

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, id):
        id = self.id_list[id]
        folder_name = "emd_" + str(id)
        map_dir = pathlib.Path(self.data_dir + "/{}".format(folder_name))
        os.chdir(map_dir)
        input_map = mrcfile.open("resampled_map.mrc", mode="r")
        input_map = deepcopy(input_map.data)
        simulated_map = mrcfile.open(
            "simulated_map_{}_res_2_vol_1.mrc".format(id), mode="r"
        )
        simulated_map = deepcopy(simulated_map.data)
        input_map = (input_map - input_map.min()) / (input_map.max() - input_map.min())
        simulated_map = (simulated_map - simulated_map.min()) / (
            simulated_map.max() - simulated_map.min()
        )

        training_transform = torchio.Compose(
            [
                torchio.RandomAnisotropy(
                    downsampling=1.5, image_interpolation="bspline", p=0.1
                ),
                torchio.RandomBlur((0, 0.5), p=0.2),
                torchio.RandomNoise(std=0.1, p=0.2),
            ]
        )
        if self.training and self.augmentation:
            input_map = torchio.ScalarImage(tensor=input_map[None, ...])
            input_map = training_transform(input_map)
            input_map = input_map.tensor.squeeze().numpy().astype(np.float32)

        original_shape = input_map.shape
        input_cube_list = np.array(
            create_cube_list(
                input_map, box_size=self.box_size, core_size=self.core_size
            )
        )
        simulated_cube_list = np.array(
            create_cube_list(
                simulated_map, box_size=self.box_size, core_size=self.core_size
            )
        )

        return (
            input_cube_list.astype(np.float32),
            simulated_cube_list.astype(np.float32),
            original_shape,
            id,
        )


class CryoEM_Map_TestDataset(torch.utils.data.Dataset):
    """
    Load the original map (normalized)
    """

    def __init__(
        self,
        data_dir,
        id_list,
        box_size=64,
        core_size=50,
    ):
        self.data_dir = data_dir
        self.id_list = id_list
        self.box_size = box_size
        self.core_size = core_size

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, id):
        id = self.id_list[id]
        folder_name = "emd_" + str(id)
        map_dir = pathlib.Path(self.data_dir + "/{}".format(folder_name))
        os.chdir(map_dir)
        input_map = mrcfile.open("resampled_map.mrc", mode="r")
        input_map = deepcopy(input_map.data)
        input_map = (input_map - input_map.min()) / (input_map.max() - input_map.min())
        original_shape = input_map.shape
        input_cube_list = np.array(
            create_cube_list(
                input_map, box_size=self.box_size, core_size=self.core_size
            )
        )

        return (
            input_cube_list.astype(np.float32),
            original_shape,
            id,
        )
