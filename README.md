# miDGD

## Setting up your environment

Create a fresh conda environment for this project

```bash
conda create -n midgd python=3.10
```

After this, activate the environment and install the requirements:

```bash
conda activate midgd
pip install -r requirements.txt
```

### Alternatives

Use mamba for seamless installation.

```bash
mamba create -n midgd python=3.10
```

Then,

```bash
mamba activate midgd

mamba install -c pytorch -c nvidia pytorch torchvision torchaudio torchmetrics pytorch-cuda=11.8 scikit-learn pandas numpy matplotlib seaborn wandb tqdm
```

## Running example code

Code for the DGD can be found in `src`. The DGD base code has for now been added to the `dgd` folder, but will be changed and imported from the Krogh group repo in the future.

An example of how to use some code has been added in the `setup_test.ipynb` notebook.

## Running miDGD

Code for the miDGD is stored in `base` and used in all python script and jupyter notebook in this repository.

The main notebook to run miDGD is the `tcga_midgd.ipynb` and the analyses is done in the `analyses.ipynb` notebook.

## Reference

The miDGD model is inspired and adapted from the Deep Generative Decoder (DGD) model, specifically scDGD [(https://doi.org/10.1093/bioinformatics/btad497)](https://doi.org/10.1093/bioinformatics/btad497) and multiDGD [(https://doi.org/10.1101/2023.08.23.554420)](https://doi.org/10.1101/2023.08.23.554420). 

The repository of the respective model is [scDGD](https://github.com/Center-for-Health-Data-Science/scDGD/) and [multiDGD](https://github.com/Center-for-Health-Data-Science/multiDGD). The minimal version of scDGD and multiDGD serve as the building block of the miDGD model.
