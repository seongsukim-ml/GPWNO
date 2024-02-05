import sys
import os
sys.path.append("..")
import json
import time

import numpy as np
import torch

# from e3nn import o3
import torch.nn.functional as F
from torch.utils.data import Dataset
# from rdkit import Chem

import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel

tqdm.pandas()
pandarallel.initialize(progress_bar=True, nb_workers=32)

# my_api_key = "pdDir9sfmf4JPMKDSJzF8W7ZJ7CyqYHE"
file_path = "../mpids_downloaded.json"
data_path = "../data"
file_pattern = ".chgcar"

with open(file_path, 'r') as file:
    mpids = json.load(file)

print("Len mpids:",len(mpids))

mpids_pd = pd.Series(mpids)

def get_stat(mpid):
    file_name = f"{mpid}{file_pattern}"
    fileobj = open(os.path.join(data_path, file_name), "r")
    readline = fileobj.readline
    readline()  # the first comment line
    scale = float(readline())  # the scaling factor (lattice constant)

    for i in range(3):
        [s for s in readline().split()]

    elements = readline().split()
    return pd.Series([mpid,elements],index=["mpid","elements"])

mpids_stat = mpids_pd.parallel_apply(get_stat)
mpids_stat.to_csv("mpids_stat_all_elements.csv")
print(mpids_stat.head())