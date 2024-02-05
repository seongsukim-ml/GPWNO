# Code of the Gaussian plane-wave neural operator.

This code is mainly inspired by the InfGCN by cheng.

## Packages
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install hydra-core lightning wandb
pip install lz4 pymatgen mp-api python-dotenv
pip install matplotlib==3.7 omegaconf
pip install e3nn
```

## Main code
```
.
├── conf                       # Training configurations
├── source                     # Source files 
│   ├── common                 
│   ├── datamodule             
│   ├── datasets               
│   └── models (✈)            # Models are here
│       └── GPWNO.py (⭐)   # Our Architecture
├── my_script                  # Shorts snippets
├── run.py                     # Main file for running the files
└── README.md
.
└── assets, configs, models, datasets # unused / Trace of InfGCN

```

## Dataset

Dataset QM9, MP, MD are not included since the size of dataset is too large.

### QM9

The QM9 dataset contains 133885 small molecules consisting of C, H, O, N, and F. The QM9 electron density dataset was
built by Jørgensen et al. ([paper](https://www.nature.com/articles/s41524-022-00863-y)) and was publicly available
via [Figshare](https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500).
Each tarball needs to be extracted, but the inner lz4 compression should be kept. We provided code to read the
compressed lz4 file.

### Cubic

The Cubic dataset contains electron charge density for 16421 (after filtering) cubic crystal system cells. The dataset
was built by Wang et al. ([paper](https://www.nature.com/articles/s41597-022-01158-z)) and was publicly available
via [Figshare](https://springernature.figshare.com/collections/Large_scale_dataset_of_real_space_electronic_charge_density_of_cubic_inorganic_materials_from_density_functional_theory_DFT_calculations/5368343).
Each tarball needs to be extracted, but the inner xz compression should be kept. We provided code to read the compressed
xz file.

**WARNING:** A considerable proportion of the samples uses the rhombohedral lattice system (i.e., primitive rhomhedral
cell instead of unit cubic cell). Some visualization tools (including `plotly`) may not be able to handle this.

### MD

The MD dataset contains 6 small molecules (ethanol, benzene, phenol, resorcinol, ethane, malonaldehyde) with different
geometries sampled from molecular dynamics (MD). The dataset was curated
from [here](https://www.nature.com/articles/s41467-020-19093-1) by Bogojeski et al.
and [here](https://arxiv.org/abs/1609.02815) by Brockherde et al. The dataset is publicly available at the Quantum
Machine [website](http://www.quantum-machine.org/datasets/).

We assume the data is stored in the `<data_root>/<mol_name>/<mol_name>_<split>/` directory, where `mol_name` should be
one of the molecules mentioned above and split should be either `train` or `test`. The directory should contain the
following files:

- `structures.npy` contains the coordinates of the atoms.
- `dft_densities.npy` contains the voxelized electron charge density data.

This is the format for the latter four molecules (you can safely ignore other files). For the former two
molecules, run `python generate_dataset.py` to generate the correctly formatted data. You can also specify the data
directory with `--root` and the output directory with `--out`.

All MD datasets assume a cubic box with side length of 20 Bohr and 50 grids per side. The densities are store as Fourier
coefficients, and we provided code to convert them.

## Materials Project

The MP dataset has not official figshare, so it need to be downloade by handed query. You can download the all MP dataset with this code:
```
python scripts/MP/download_mp.py
```

Note that the size of the MP dataset is about 10T. We assume the data is stored in the `../dataset_mp_mixed` with the data split we attach.

## Running the code

Before run the code, you have to modify the `.env` files as proper directory in your system.

### MD
```
python run.py --config-name=md \
    model=GPWNO \
    data.datamodule.datasets.test.n_samples=Null \
    logging=draw \
    model.model_sharing=False \
    data.mol_name=benzene
```

### QM9
```
python run.py --config-name=qm9 \
    model=GPWNO \
    data.train_max_iter=80000 \
    data.num_test_samples=1600 \
    logging=draw \
```

### MP
```
python run.py -m \
    --config-name=mp_mixed \
    model=GPWNO \
    data=mp_tetragonal \
    data.num_workers=32 \
    logging=draw
```