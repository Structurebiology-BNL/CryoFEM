"""
the script is for using ChimeraX to resample the average map (from two half-maps)
and generate target maps used for training and evaluating the ResEM model.
it requires the input map and its corresponding PDB model.
Change the CHIMERAX, OUTPUT_RESAMPLED_MAP and INPUT_MAP paths
"""
from tqdm import tqdm
import pathlib
import os
import subprocess
import json

with open("sample_emd_ids.json", "r") as f:
    ids = json.load(f)

CHIMERAX_PATH = "/usr/bin/chimerax"
OUTPUT_RESAMPLED_MAP_PATH = "/data/emdb_data/"
INPUT_MAP_PATH = "/data/emdb_half_maps/"

print("total maps to process: ", len(ids))
failed_list = []
keys = list(ids.keys())
for id in tqdm(keys):
    folder_name = "emd_" + id
    map_dir = pathlib.Path(OUTPUT_RESAMPLED_MAP_PATH + "{}".format(folder_name))
    map_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(map_dir)
    print("id:", id)
    map = pathlib.Path("resampled_map.mrc")
    if map.is_file():
        continue
    try:
        input = INPUT_MAP_PATH + "{}/averaged_map_{}.ccp4".format(folder_name, id)
        pdb_model = ids[id] 
        pdb_file = INPUT_MAP_PATH + "{}/{}".format(folder_name, pdb_model)
        output_path = "simulated_map_{}_res_2_vol_1.mrc".format(id)
        with open("resample_and_simulation.cxc", "w") as chimerax_script:
            chimerax_script.write(
                "open " + input + "\n"
                "vol resample #1 spacing 1.0 gridStep 1\n"
                "vol #2 step 1\n"
                "save resampled_map.mrc model #2\n"
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
    with open("failed_resampled_simulated_emd_id.txt", "w") as outfile:
        outfile.write("\n".join(failed_list))
