from torchvision.datasets.utils import download_url
from tqdm import tqdm
import json
import pathlib
import pandas as pd
import argparse


def fasta_pdb_download(batch, total_batches=5):
    with open("/host/ResEM/data_processing/emdb_ids_full.txt") as file:
        ids = file.readlines()
        ids = [line.rstrip() for line in ids]

    batch_size = len(ids) // total_batches
    ids_ = ids[batch * batch_size : (batch + 1) * batch_size]

    failed_list = []
    emd_to_pdb = {}
    for id in tqdm(ids_):
        id = id[4:]
        folder_name = "emd_" + id
        emdb_link = "https://www.ebi.ac.uk/emdb/EMD-{}?tab=links".format(id)
        try:
            new_dir = pathlib.Path("/host/full_emd_data/{}".format(folder_name))
            new_dir.mkdir(parents=True, exist_ok=True)
            tables = pd.read_html(emdb_link)  # Returns list of all tables on page
            pdb_table = tables[0]  # Select table of interest
            pdb_id = pdb_table["Accession"].values[0].upper()
            emd_to_pdb[id] = pdb_id
            print(id, pdb_id)
            fasta_link = "https://www.rcsb.org/fasta/entry/{}".format(pdb_id)
            download_url(
                fasta_link, "/host/full_emd_data", filename="{}.fasta".format(pdb_id)
            )
            pdb_link = "https://files.rcsb.org/download/{}.pdb".format(pdb_id)
            download_url(pdb_link, new_dir)

        except:
            print("Download failed for ", id)
            failed_list.append(id)

    with open("emd_to_pdb_{}.json".format(batch), "w") as f:
        json.dump(emd_to_pdb, f, indent=4)

    if len(failed_list) > 0:
        with open("failed_ids_{}.txt".format(batch), "w") as f:
            for line in failed_list:
                f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download fasta and pdb files")
    parser.add_argument("--batch", type=int, help="batch number to download.")
    args = parser.parse_args()
    fasta_pdb_download(args.batch)
