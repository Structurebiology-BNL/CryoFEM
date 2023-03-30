from torchvision.datasets.utils import download_url
import json
import pathlib
from tqdm import tqdm
import argparse


def download_halfmaps(batch, total_batches=7):
    with open("/host/ResEM/data_processing/emdb_id_30_identity.json", "r") as f:
        ids = json.load(f)
    assert batch < total_batches, "Batch number should be less than total batches."
    ids = list(ids.keys())
    batch_size = len(ids) // total_batches
    if batch == total_batches - 1:
        ids = ids[batch * batch_size :]
    else:
        ids = ids[batch * batch_size : (batch + 1) * batch_size]
    failed_list = []
    for id in tqdm(ids):
        folder_name = "emd_" + id
        link_1 = "https://ftp.wwpdb.org/pub/emdb/structures/EMD-{}/other/emd_{}_half_map_1.map.gz".format(
            id, id
        )
        link_2 = "https://ftp.wwpdb.org/pub/emdb/structures/EMD-{}/other/emd_{}_half_map_2.map.gz".format(
            id, id
        )
        try:
            new_dir = pathlib.Path("/host/full_emd_data/{}".format(folder_name))
            new_dir.mkdir(parents=True, exist_ok=True)
            download_url(link_1, new_dir)
            download_url(link_2, new_dir)

        except:
            print("Download failed for ", id)
            failed_list.append(id)

    if len(failed_list) > 0:
        with open("failed_half_map_emd_id_{}.json".format(batch), "w") as f:
            json.dump(failed_list, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EM maps")
    parser.add_argument("--batch", type=int, help="batch number to download.")
    args = parser.parse_args()
    download_halfmaps(args.batch)
