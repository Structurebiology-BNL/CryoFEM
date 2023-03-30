"""
the script is using ChimeraX to resample the average map (from two half-maps)
and generate target maps used for training and evaluating the ResEM model.
It requires the input map and its corresponding PDB model.
Please change the CHIMERAX, OUTPUT_RESAMPLED_MAP and INPUT_MAP paths accordingly.
"""
import pathlib
import os
import subprocess
import json
import argparse


CHIMERAX_PATH = "/usr/bin/chimerax"
OUTPUT_RESAMPLED_MAP_PATH = "/host/full_emd_data/"
INPUT_MAP_PATH = "/host/full_emd_data/"


def map_resample_and_simulation(batch, total_batches=5):
    assert batch < total_batches, "Batch number should be less than total batches."
    # load the id list which excludes the current training set
    with open("final_valid_ids_with_pdb_exclude_current_ones.json", "r") as f:
        ids = json.load(f)

    id_list = list(ids.keys())
    failed_list = []
    batch_size = len(id_list) // total_batches
    if batch == total_batches - 1:
        id_list = id_list[batch * batch_size :]
    else:
        id_list = id_list[batch * batch_size : (batch + 1) * batch_size]

    print("total maps to process: ", len(id_list))
    for id in id_list:
        folder_name = "emd_" + id
        path = pathlib.Path(INPUT_MAP_PATH + "{}".format(folder_name))
        os.chdir(path)
        print("id:", id)
        resampled_map = "resampled_map_{}.mrc".format(id)
        map = pathlib.Path(resampled_map)
        output_path = "simulated_map_{}_res_2_vol_1.mrc".format(id)
        simulated_map = pathlib.Path(output_path)
        if map.is_file() and simulated_map.is_file():
            continue
        try:
            input = INPUT_MAP_PATH + "{}/averaged_map_{}.mrc".format(folder_name, id)
            pdb_model = ids[id]
            pdb_file = INPUT_MAP_PATH + "{}/{}.pdb".format(folder_name, pdb_model)
            with open("resample_and_simulation.cxc", "w") as chimerax_script:
                chimerax_script.write(
                    "open " + input + "\n"
                    "vol resample #1 spacing 1.0 gridStep 1\n"
                    "vol #2 step 1\n"
                    "save " + resampled_map + " model #2\n"
                    "open " + pdb_file + "\n"
                    "molmap #3 2 onGrid #2 \n"
                    "save " + output_path + " format mrc model #4\n"
                    "exit"
                )
        except:
            print("Resampling and simulation failed for ", id)
            failed_list.append(id)
        try:
            # Changed the chimera path to an input
            subprocess.run([CHIMERAX_PATH, "--nogui", chimerax_script.name])
            os.remove(chimerax_script.name)
        except:
            print("Simulation failed for ", id)
            failed_list.append(id)

    if len(failed_list) > 0:
        with open("failed_resampled_simulated_id_{}.txt".format(batch), "w") as outfile:
            outfile.write("\n".join(failed_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample and simulate the map using ChimeraX."
    )
    parser.add_argument("--batch", type=int, help="batch number to process.")
    args = parser.parse_args()
    map_resample_and_simulation(args.batch)
