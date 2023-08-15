### AlphaFold model
The simplest way to get the AlphaFold prediction is using [ColabFold](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb).
* **Initial prediction**
    * Use the `pdb100` (updated from `pdb70`) database for the template search
    * Use the default settings for MSA and Amber-relaxation.
* **Rebuilding with Phenix**
    * After rebuilding the model with Phenix (see below), supply the rebuilt model as a template for the second round of prediction.
    * Set `template_mode=custom`` and upload the rebuilt model in the ColabFold notebook.

* **Custom Template Runs:**
    * Conduct two runs with or without MSA when using the custom template.
    * Proceed with the one with higher correlations with the raw map.
    * Turn off the MSA by setting `msa_mode=single_sequence`` in the ColabFold notebook.
### Phenix model building
This is an example script of using `phenix.dock_and_rebuild` to refine a model against the (enhanced) cryo-EM map
```bash
phenix.dock_and_rebuild model=your_protein_model.pdb full_map=enhanced_map.mrc\
   resolution=2.0\ # you can try using the reported resolution of the map to see which one gives higher correlation with the map
   nproc=4\ # number of processors to use
   maximum_domains=3 # for larger chains, this can be increased
```

### Model quality evaluation
When the refinement is done, you can check the model quality using the following commands:
* Real space map-model correlations
    ```bash
    phenix.map_model_cc resolution=resolution ignore_symmetry_conflicts=True pdb_file map_file
    ```

* Map-model FSC
    ```bash
    phenix.mtriage map_file pdb_fil
    ```
* Chain comparison
    * In case the ground truth model is available e.g., deposited PDB, you can also use the following command to evaluate the model quality:
        ```bash
        phenix.superpose_and_morph fixed_model=ground_truth_model.pdb moving_model=your_own_model.pdb trim=False 

        phenix.chain_comparison max_dist=2 ground_truth_model.pdb
        your_own_model_superposed.pdb  
        ```
    * Here `max_dist` is the maximum distance between two $C_{\alpha}$ atoms in the two models to be considered as a match. The default value is 3 Å.