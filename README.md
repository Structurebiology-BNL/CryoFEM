## Introduction
This is the official implementation of ResEM (**Res**olution **E**nhance**M**ent)[[preprint](https://www.biorxiv.org/content/early/2023/02/03/2023.02.02.526877)], an image enhancement tool based on 3D convolutional neural networks for cryo-EM density map post-processing. It can effectively increase the resolution (calculated from FSC) of the density map hence facilitate better map interpretation at the atomic level.


Our image enhancement model can be used synergistically with [AlphaFold](https://github.com/deepmind/alphafold) and protein model refinement tools, e.g., [PHENIX](https://phenix-online.org/), to tackle cases where initial AlphaFold predictions are less accurate.

<img src="https://github.com/Structurebiology-BNL/ResEM/blob/main/utils/flow_chart.png" width=45% height=45%>

In the manuscript we used our forked version of [OpenFold](https://github.com/empyriumz/openfold), which enables us to use custom template to perform structural predictions. Alternatively, one can use ColabFold [official](https://github.com/sokrypton/ColabFold) to implement the proposed workflow. 

## Requirements
### Update
* March 2023: Try our Colab notebook to run ResEM on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Structurebiology-BNL/ResEM/blob/main/Colab_ResEM.ipynb). Simply upload your own half maps and run the prediction, then you can see the visualization and download the enhanced map. Check the [Colab notebook](https://colab.research.google.com/github/Structurebiology-BNL/ResEM/blob/main/Colab_ResEM.ipynb) for more details.
  
* March 2023: We have tested ResEM with PyTorch 2.0. By default if you run the training script with PyTorch 2.0, it will first compile to model with `torch.compile` to accelerate the training

ResEM is developed with Python 3.9 and PyTorch 1.12. Other important packages include:
```
torchio 0.18.86
scikit-image 0.19.3 
mrcfile 1.4.3
numpy 1.24.1
tqdm 4.64.1
```
<!-- where [torchio](https://torchio.readthedocs.io/) is used for data augmentation in the training, [gemmi](https://gemmi.readthedocs.io/en/latest/) is for averaging half-maps,  -->
We recommend to set up a new [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) environment to install those Python packages.
* 3rd party dependency: we use [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) to perform map resampling (making each voxel has the dimension of 1<span>&#8491;</span>) and target generation (simulate target maps using deposited PDB models).

## Usage

### Data collection
All the data used in the training and validation of ResEM are publicly available from [PDB](https://www.rcsb.org/) and [EMDB](https://www.emdataresource.org/). `data_processing/train_emd_id.txt` contains the list of EMDB ID list used for model training.

* After downloading the half-maps from EMDB, using `data_processing/generate_averaged_map_from_half_maps.py` to compute and save the averaged raw map.
* Checkout `data_processing/map_resampling_simulation.py` to see how to resample the raw map and simulate the target map using ChimeraX. 

### Batch Inference
1. Download our trained model from [Google drive](https://drive.google.com/file/d/1hCaEbYxQV56JIpN2c2iJSiiKAgRu7TT6/view?usp=sharing)
2. We provide a sample data at `example_data` and the corresponding inference configuration at `configs/inference.json`. After tuning the options, e.g., trained weights path, GPU id, run the inference script as:
```
python inference.py --config configs/inference.json 
```
3. If `{"test_data": "save_output"=1}`, the output maps will be saved to `./results/inference/yyyy-mm-dd-current-clock-time`. It would take around 10s using the sample map on a Nvidia V100 GPU.

### Training
1. In addition to the resampled raw map, you'll need simulated maps as the targets to train the model. By default, `data_processing/map_resampling_simulation.py` will save the simulated map as `simulated_map_{xxx}_res_2_vol_1.mrc`, where `{xxx}` denotes the EMDB ID, `res_2` and `vol_1` indicate the simulated resolution of 2 <span>&#8491;</span> and voxel size of 1 <span>&#8491;</span>, respectively.
2. We provide a sample training configuration file at `configs/train.json`. 
3. After tuning the options, e.g. GPU id, batch size, # of epochs, run the training script as follows:
```
python train_model.py --config configs/train.json
```
3. Depending on the options in `train.json`, the training configuration, log, and/or trained models will be saved to `./results/train/yyyy-mm-dd-current-clock-time`.

### Cite
If you find our work helpful, please consider cite our work as follows:
```
@article {resem2023,
	author = {Dai, Xin and Wu, Longlong and Yoo, Shinjae and Liu, Qun},
	title = {Integrating AlphaFold and deep learning for atomistic interpretation of cryo-EM maps},
	year = {2023},
	doi = {10.1101/2023.02.02.526877},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/02/03/2023.02.02.526877},
	journal = {bioRxiv}
}
```

### License
This source code is licensed under the CSI approved 3-clause BSD license found in the LICENSE file in the root directory of this source tree.