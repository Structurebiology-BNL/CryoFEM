"""
getting from https://github.com/DrDongSi/Ca-Backbone-Prediction/blob/f921e971ac2bb6e70844619f0f420a6f34dabeba/cnn/map_splitter.py
This files contains functions used to split a large protein into smallers 64^3 chunks that can
fit into the training CNN. This is accomplished without any input from the user. When the output
image is reconstructured only the middle 50^3 region in the image is used to build the output
protein. This helps eliminate the issue of boundary prediction issues.
Moritz, Spencer  November, 2018

args:
box_size: Expected CNN dimensionality
core_size: Core of the image where we don't need to worry about boundary issues.

"""
import numpy as np
import math


def get_manifest_dimensions(image_shape, core_size=50):
    dimensions = [0, 0, 0]
    dimensions[0] = math.ceil(image_shape[0] / core_size) * core_size
    dimensions[1] = math.ceil(image_shape[1] / core_size) * core_size
    dimensions[2] = math.ceil(image_shape[2] / core_size) * core_size
    return dimensions


# Creates a list of 64^3 tensors. Each tensor can be fed to the CNN independently.
def create_manifest(full_image, box_size=64, core_size=50):
    image_shape = np.shape(full_image)
    padded_image = np.zeros(
        (
            image_shape[0] + 2 * box_size,
            image_shape[1] + 2 * box_size,
            image_shape[2] + 2 * box_size,
        )
    )
    padded_image[
        box_size : box_size + image_shape[0],
        box_size : box_size + image_shape[1],
        box_size : box_size + image_shape[2],
    ] = full_image

    manifest = []

    start_point = box_size - int((box_size - core_size) / 2)
    cur_x = start_point
    cur_y = start_point
    cur_z = start_point
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        next_chunk = padded_image[
            cur_x : cur_x + box_size, cur_y : cur_y + box_size, cur_z : cur_z + box_size
        ]
        manifest.append(next_chunk)
        cur_x += core_size
        if cur_x + (box_size - core_size) / 2 >= image_shape[0] + box_size:
            cur_y += core_size
            cur_x = start_point  # Reset
            if cur_y + (box_size - core_size) / 2 >= image_shape[1] + box_size:
                cur_z += core_size
                cur_y = start_point  # Reset
                cur_x = start_point  # Reset
    return manifest


# Takes the output of the CNN and reconstructs the full dimensionality of the protein.
def reconstruct_maps(manifest, image_shape, box_size=64, core_size=50):
    extract_start = int((box_size - core_size) / 2)
    extract_end = int((box_size - core_size) / 2) + core_size
    dimensions = get_manifest_dimensions(image_shape, core_size=core_size)

    reconstruct_image = np.zeros((dimensions[0], dimensions[1], dimensions[2]))
    counter = 0
    for z_steps in range(int(dimensions[2] / core_size)):
        for y_steps in range(int(dimensions[1] / core_size)):
            for x_steps in range(int(dimensions[0] / core_size)):
                reconstruct_image[
                    x_steps * core_size : (x_steps + 1) * core_size,
                    y_steps * core_size : (y_steps + 1) * core_size,
                    z_steps * core_size : (z_steps + 1) * core_size,
                ] = manifest[counter][
                    extract_start:extract_end,
                    extract_start:extract_end,
                    extract_start:extract_end,
                ]
                counter += 1
    reconstruct_image = np.array(reconstruct_image, dtype=np.float32)
    reconstruct_image = reconstruct_image[
        : image_shape[0], : image_shape[1], : image_shape[2]
    ]
    return reconstruct_image
