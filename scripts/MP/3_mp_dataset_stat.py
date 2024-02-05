import pandas as pd
import numpy as np
from pymatgen.io.vasp.outputs import Chgcar as chg
import os
from tqdm import tqdm
import json
from pandarallel import pandarallel

tqdm.pandas()
pandarallel.initialize(progress_bar=True, nb_workers=32)

# my_api_key = "pdDir9sfmf4JPMKDSJzF8W7ZJ7CyqYHE"
# file_path = "../mpids_downloaded.json"
# file_path = "../mpids_all.json"

# with open(file_path, 'r') as file:
#     mpids = json.load(file)

chgcar_path = "../data"
saved_files = os.listdir(chgcar_path)
saved_files_wo_ext = []
for i in saved_files:
    saved_files_wo_ext.append(i.split(".")[0])

mpids = saved_files_wo_ext

print("Len mpids:",len(mpids))

mpids_pd = pd.Series(mpids)

def get_stat(mpid):
    try:
        file_path = "../data/"+mpid+".chgcar"
        chgcar = chg.from_file(file_path)
        struc = chgcar.structure
        a,b,c = struc.lattice.abc
        grid_x,grid_y,grid_z = chgcar.dim
        n_atoms = struc.num_sites
    except:
        a,b,c = np.nan,np.nan,np.nan
        grid_x,grid_y,grid_z = np.nan,np.nan,np.nan
        n_atoms = np.nan

    return pd.Series([mpid,a,b,c,(a+b+c)/3,grid_x,grid_y,grid_z,(grid_x*grid_y*grid_z)**(1/3), n_atoms],
                     index=["mpid","a","b","c","avg","grid_x","grid_y","grid_z","grid","n_atoms"]
                     )

mpids_stat = mpids_pd.parallel_apply(get_stat)
mpids_stat.to_csv("mpids_stat_all.csv")
print(mpids_stat.head())