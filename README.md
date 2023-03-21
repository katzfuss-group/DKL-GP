## Overview
Repository for [Variational sparse inverse Cholesky approximation for latent Gaussian processes via double Kullback-Leibler minimization](https://arxiv.org/abs/2301.13303)

## Local modules
Local modules, other than our main algorithm, are stored in the `external` folder. To install everything together, you can run `source INSTALL`.

## Our method 
Our methods are implemented in the `viva` module.

## Experiments 
Our experiments are stored in the `experiments` folder, some of which are controled by the `setups.yaml` file.

For a simple demonstration, you can run `example_1D.ipynb` directly after running the installation script. 

When running `main.py`:

  - to produce the results in our simulation study, you need to first run `data_gen.py` to produce simulated datasets. 
  - for results in the real-data study, you need to download the real datasets into the `datasets` folder and split them into `train` and `test`. An example is given with the `elevators` dataset. You may also ask the authors for the real datasets to reproduce the same results. 
