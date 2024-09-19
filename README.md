<p>
  <img src="https://img.shields.io/badge/python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/pyg-3C2179?style=flat-square&logo=pyg&logoColor=white"/>
  <img src="https://img.shields.io/badge/lightning-792EE5?style=flat-square&logo=lightning&logoColor=white"/>
</p>

# Gaussian plane-wave neural operator (GPWNO) [ICML 2024].
By Seongsu Kim, Feb, 2024 [[arxiv]](https://arxiv.org/abs/2402.04278) [[PDF]](https://arxiv.org/pdf/2402.04278.pdf)

🌟 This repository contains an implementation of the ICML 2024 paper ***Gaussian plane-wave neural operator for electron density estimation***. The code implementation of the model is mainly inspired by the [InfGCN](https://github.com/ccr-cheng/infgcn-pytorch) by Chaoran Cheng.

## Packages and Requirements
All codes are run with python 3.9 and CUDA 12.0. A similar environment should also work, as this project does not rely on some rapidly changing packages.

```
conda create -n GPWNO python==3.9
conda activate GPWNO

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install numpy==1.23.4 scipy==1.9.3 matplotlib==3.7 tqdm
pip install hydra-core lightning wandb omegaconf
pip install lz4 pymatgen mp-api python-dotenv PyYAML easydict
pip install e3nn
```

## Directory and Files
```
.
├── conf                        # Training configurations
├── source                      # Source files 
│   ├── baseline (optional)             
│   ├── common                 
│   ├── datamodule             
│   ├── datasets               
│   └── models (✈)              # Models are here
│       └── GPWNO.py (⭐)       # Our Architecture
├── scripts                     # Shorts snippets
├── run.py                      # Main file for running the files
└── README.md
```


---
## Dataset

Datasets QM9, MP, and MD are not included in the repository since the size of the datasets is too large.

### QM9

The QM9 dataset contains 133885 small molecules consisting of C, H, O, N, and F. The QM9 electron density dataset was
built by Jørgensen et al. ([paper](https://www.nature.com/articles/s41524-022-00863-y)) and was publicly available
via [Figshare](https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500).
Each tarball needs to be extracted, but the inner lz4 compression should be kept. We provided code to read the
compressed lz4 file.

For convenience, we provide the list of the ```scripts\qm9_url.txt``` that splits url list of the QM9 files.
After download ```aria2c```, You can use the below command
```
aria2c --input-file scripts\qm9_url.txt -d ../dataset_qm9
```
You can additionally use the ```-x``` keywords to set the multiple connections.
See [aria2](https://aria2.github.io/) for more information.

### MD

The MD dataset contains 6 small molecules (ethanol, benzene, phenol, resorcinol, ethane, malonaldehyde) with different
geometries sampled from molecular dynamics (MD). The dataset was curated
from [here](https://www.nature.com/articles/s41467-020-19093-1) by Bogojeski et al.
and [here](https://arxiv.org/abs/1609.02815) by Brockherde et al. The dataset is publicly available at the Quantum
Machine [website](http://www.quantum-machine.org/datasets/).

We assume the data is stored in the `../dataset/<mol_name>/<mol_name>_<split>/` directory, where `mol_name` should be
one of the molecules mentioned above and split should be either `train` or `test`. The directory should contain the
following files:

- `structures.npy` contains the coordinates of the atoms.
- `dft_densities.npy` contains the voxelized electron charge density data.

This is the format for the latter four molecules (you can safely ignore other files). For the former two
molecules, run `python generate_dataset.py` to generate the correctly formatted data. You can also specify the data
directory with `--root` and the output directory with `--out`.

All MD datasets assume a cubic box with a side length of 20 Bohr and 50 grids per side. The densities are stored as Fourier
coefficients, and we provided code to convert them.

### Materials Project

The [MP dataset](https://next-gen.materialsproject.org/ml/charge_densities) curated from [here](https://arxiv.org/abs/2107.03540) is the dataset of the inorganic crystalline materials. We newly conduct a benchmark from the MP by categorizing the 117,535 molecules into seven crystal families:  triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, and cubic. We also providied the list of the materials on our experiments in the code. 

MP does not have an official download site,
it needs to be downloaded by handed query by API. To download the full dataset, you can use the download code we provide. You need to register your API key in
```scirpts/MP/download_mp.py``` from [Materials project](https://next-gen.materialsproject.org/api).

You can download all the MP datasets with this code
```
python scripts/MP/download_mp.py
```

Note that the size of the MP dataset is about 10T. We assume the data is stored in the `../dataset_mp` with the data split we attach.
FYI, it took about 7 days to download all for me.

---
## Running the code

**\[⚠️Note\]** Before running the code, you have to modify the `.env` files as a proper directory in your system.
And also you have to fix the path of the configure in the `conf/data`. Other scripts that is used for running code are in `scripts` directory.

### MD
```
# can specify the molecular name
python run.py --config-name=md \
    model=GPWNO_MD \
    data.num_workers=32 \
    logging=draw \
    data.mol_name=benzene
```

### QM9
```
python run.py --config-name=qm9 \
    model=GPWNO_QM9
```

### MP
```
# default is mixed
python run.py --config-name=mp \
    model=GPWNO_pbc

# specify the lattice type
python run.py --config-name=mp \
    model=GPWNO_pbc \
    data=mp_tetragonal
```

## Citation
If you find this code useful, please cite our paper
```
@misc{kim2024gaussian,
      title={Gaussian Plane-Wave Neural Operator for Electron Density Estimation}, 
      author={Seongsu Kim and Sungsoo Ahn},
      year={2024},
      eprint={2402.04278},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}
```

## Baseline reference
We implemented these baselines in our repo:

[DeepDFT](https://github.com/peterbjorgensen/DeepDFT)

[InfGCN](https://github.com/ccr-cheng/infgcn-pytorch)

[charge3net](https://github.com/AIforGreatGood/charge3net)
