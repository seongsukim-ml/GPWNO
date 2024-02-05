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

# from torch_geometric.data import Data
path_dir = "../data"

saved_files = os.listdir(path_dir)
saved_files_wo_ext = []
for i in saved_files:
    if i.split(".")[1] == "chgcar":
        saved_files_wo_ext.append(i.split(".")[0])

item = 0
file_pattern = ".chgcar"
file_list = saved_files_wo_ext
print(file_list[item])

file_name = f"{(file_list[item])}{file_pattern}"
data_path = path_dir

fileobj = open(os.path.join(data_path, file_name), "r")

import multiprocessing

error = []


def process_file(jj):
    try :
        file_name = f"{(file_list[jj])}{file_pattern}"
        fileobj = open(os.path.join(data_path, file_name), "r")
        readline = fileobj.readline
        readline()  # the first comment line

        scale = float(readline())  # the scaling factor (lattice constant)

        for i in range(3):
            [s for s in readline().split()]

        elements = readline().split()
        n_atoms = [int(s) for s in readline().split()]

        readline()
        tot_atoms = sum(n_atoms)
        atom_coord = torch.empty(tot_atoms, 3, dtype=torch.float)

        idx = 0
        for elem, n in zip(elements, n_atoms):
            for _ in range(n):
                [s for s in readline().split()]
                idx += 1

        readline()  # an empty line
        shape = [int(s) for s in readline().split()]  # grid size
        n_grid = shape[0] * shape[1] * shape[2]

        density = len([s for s in fileobj.read().split()[:n_grid]])
        # print("mpid: ", file_list[jj])
        # print("density: ", density)
        # print("grid shape: ", shape, shape[0] * shape[1] * shape[2])
        if density != shape[0] * shape[1] * shape[2]:
            print("[!] density shape not match", jj, file_list[jj])
            print("density: ", density)
            print("grid shape: ", shape, shape[0] * shape[1] * shape[2])
            error.append(file_list[jj])
    except ValueError:
        print("[!] ValueError", jj, file_list[jj])
        error.append(file_list[jj])
        return


# Create a pool of processes
pool = multiprocessing.Pool(32)

# Map the process_file function to each value of jj using the pool of processes
pool.map(process_file, range(len(saved_files_wo_ext)))

# Close the pool of processes
pool.close()

with open("error.json", "w") as f:
    json.dump(error, f)
