"""
this script uses mrcfile to read the half-maps then write the 
averaged map
"""
import mrcfile
import os
import pathlib
from tqdm import tqdm
from copy import deepcopy
import argparse

INPUT_MAP_PATH = "/host/full_emdb_data/"


def average_map(batch, total_batches=7):
    with open("final_valid_ids.txt") as file:
        ids = file.readlines()
        ids = [line.rstrip() for line in ids]
    assert batch < total_batches, "Batch number should be less than total batches."
    batch_size = len(ids) // total_batches
    if batch == total_batches - 1:
        ids = ids[batch * batch_size :]
    else:
        ids = ids[batch * batch_size : (batch + 1) * batch_size]
    failed_list = []
    for id in tqdm(ids):
        folder_name = "emd_" + id
        print(id)
        path = pathlib.Path(INPUT_MAP_PATH + "{}".format(folder_name))
        os.chdir(path)
        try:
            m1 = mrcfile.open("emd_{}_half_map_1.map.gz".format(id), mode="r")
            meta_data = deepcopy(m1.header)
            m2 = mrcfile.open("emd_{}_half_map_2.map.gz".format(id), mode="r")
            average = 0.5 * (m1.data + m2.data)
            with mrcfile.new("averaged_map_{}.mrc".format(id)) as mrc:
                mrc.set_data(average)
                mrc.header.cella.x = meta_data.cella.x
                mrc.header.cella.y = meta_data.cella.y
                mrc.header.cella.z = meta_data.cella.z
                mrc.header.nxstart = meta_data.nxstart
                mrc.header.nystart = meta_data.nystart
                mrc.header.nzstart = meta_data.nzstart
        except:
            print("Averaging failed for ", id)
            failed_list.append(id)

    if len(failed_list) > 0:
        with open("failed_id_for_average_{}.txt".format(batch), "w") as outfile:
            outfile.write("\n".join(failed_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average half maps.")
    parser.add_argument("--batch", type=int, help="batch number to process.")
    args = parser.parse_args()
    average_map(args.batch)
