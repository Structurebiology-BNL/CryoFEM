"""
this script uses gemmi to read the half-maps then write the 
averaged map
"""
import gemmi
import numpy as np
import os
import pathlib
from tqdm import tqdm
import json

INPUT_MAP_PATH = "/data/emdb_half_maps/"

with open("sample_emd_ids.json", "r") as f:
    ids = json.load(f)
    
failed_list = []
for id in tqdm(ids):
    folder_name = "emd_" + id
    print(id)
    path = pathlib.Path(INPUT_MAP_PATH + "{}".format(folder_name))
    os.chdir(path)
    my_file = pathlib.Path("averaged_map_{}.ccp4".format(id))
    try:
        m1 = gemmi.read_ccp4_map("emd_{}_half_map_1.map.gz".format(id))
        m1_arr = np.array(m1.grid, copy=False)
        m2 = gemmi.read_ccp4_map("emd_{}_half_map_2.map.gz".format(id))
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
        ccp4.write_ccp4_map("averaged_map_{}.ccp4".format(id))
    except:
        print("Averaging failed for ", id)
        failed_list.append(id)

if len(failed_list) > 0:
    with open("failed_resampled_simulated_emd_id.txt", "w") as outfile:
        outfile.write("\n".join(failed_list))
