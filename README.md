## Introduction
ResEM(**Res**olution **E**nhance**M**ent) is an image enhancement tool based on 3D convolutional neural networks for cryo-EM density map post-processing. It can effectively increase the resolution (calculated from FSC) of the density map hence facilitate better map interpretation at the atomic level.


## Requirements
ResEM is developed with Python 3.9 and PyTorch 1.12. Other important packages include:
```
torchio
gemmi
skimage
mrcfile
```
<!-- where [torchio](https://torchio.readthedocs.io/) is used for data augmentation in the training, [gemmi](https://gemmi.readthedocs.io/en/latest/) is for averaging half-maps,  -->

* 3rd party depency: we use [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) to perform map resampling (making each voxel has the dimension of 1<span>&#8491;</span>) and target simulation (generate target maps using deposited PDB models)

## Usage

### Data collection
All the data used in the training and validation of ResEM are publicly available from [PDB](https://www.rcsb.org/) and [EMDB](https://www.emdataresource.org/)

* After downloading the half-maps from EMDB, using `data_processing/generate_averaged_map_from_half_maps.py` to compute and save the averaged raw map.
* Checkout `data_processing/map_resampling_simulation.py` to see how to resample the raw map and simulate the target map using ChimeraX.

### Inference
1. Download our trained model from [Google drive](https://drive.google.com/file/d/1hCaEbYxQV56JIpN2c2iJSiiKAgRu7TT6/view?usp=sharing)
2. We provide a sample inference configurations in `configs/inference.json`. After tuning the options, e.g., input data path, trained weights path, GPU id, run the inference script as:
```
python inference.py --inference_config configs/inference.json 
```
3. If `"test_data": {"save_output"=1}`, the output maps will be saved to `./results/inference/yyyy-mm-dd`.

### Training
1. In addition to the resampled raw map, you'll need simulated maps as the targets to train the model. By default, `data_processing/map_resampling_simulation.py` will save the simulated map as `simulated_map_{xxx}_res_2_vol_1.mrc`, where `{xxx}` denotes the EMDB ID, `res_2` and `vol_1` indicate the simulated resolution of 2 <span>&#8491;</span> and voxel size of 1 <span>&#8491;</span>, respectively.
2. You'll find a sample training configuration file at `configs/train.json`. 

```
python train_model.py --config configs/train.json
```
3. Depending on the training options, the training logs, configurations, and/or trained model will be saved to `./results/train/yyyy-mm-dd`.

### Cite
If you find our work helpful, please consider cite our work as follows: