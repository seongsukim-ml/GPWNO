# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import os
import tarfile
import time
from pathlib import Path
from typing import Union, Optional

from functools import partial

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from hydra.utils import instantiate

from source.datasets.charge3net.collate import collate_list_of_dicts
from source.datasets.charge3net.split import split_data
import source.datasets.charge3net.data as utils
from source.datasets.charge3net.graph_construction import GraphConstructor
import json


class DistributedEvalSampler(torch.utils.data.Sampler):
    """Distributed sampler that does not add duplicates or drop last like DistributedSampler, for testing only"""

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            world = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.world = world
        self.rank = rank
        self.total_size = len(self.dataset)
        indices = list(range(self.total_size))
        indices = indices[self.rank : self.total_size : self.world]
        self.num_samples = len(indices)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank : self.total_size : self.world]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DensityDatamodule:
    def __init__(
        self,
        root: Union[str, bytes, os.PathLike],
        split_file: Optional[Union[str, bytes, os.PathLike]],
        graph_constructor: GraphConstructor,
        num_probes: int = None,  # used to set for all probes
        train_probes: int = 1024,
        val_probes: int = 1024,
        test_probes: int = None,
        batch_size: int = 32,
        train_workers: int = 16,
        val_workers: int = 16,
        pin_memory: bool = False,
        # val_frac: float = 0.005,
        drop_last: bool = False,
        grid_size_file: Optional[Union[str, bytes, os.PathLike]] = None,
        max_grid_construction_size: int = 1000000,
        **kwargs,
    ):

        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.pin_memory = pin_memory
        # self.val_frac = val_frac
        self.split_file = split_file
        self.grid_size_file = grid_size_file
        self.max_grid_construction_size = max_grid_construction_size
        self.drop_last = drop_last

        if num_probes is not None:
            train_probes = val_probes = test_probes = num_probes

        # self.train_gc = graph_constructor(num_probes=train_probes)
        # self.val_gc = graph_constructor(num_probes=val_probes)
        # self.test_gc = graph_constructor(num_probes=test_probes)

        self.train_gc = instantiate(graph_constructor, num_probes=train_probes)()
        self.val_gc = instantiate(graph_constructor, num_probes=val_probes)()
        self.test_gc = instantiate(graph_constructor, num_probes=test_probes)()
        # import pdb

        # pdb.set_trace()

        # dataset = DensityData(self.data_root)
        # subsets = split_data(
        #     dataset, val_frac=self.val_frac, split_file=self.split_file
        # )
        # self.train_set = DensityGraphDataset(subsets["train"], self.train_gc)
        # self.val_set = DensityGraphDataset(subsets["validation"], self.val_gc)
        # self.test_set = DensityGraphDataset(
        #     subsets["test"],
        #     self.test_gc,
        #     grid_size_file=self.grid_size_file,
        #     max_grid_size=self.max_grid_construction_size,
        # )
        self.dataset = {}
        self.dataset["train"] = DensityDataDirSplit(
            root=self.root,
            split="train",
            split_file=self.split_file,
        )
        self.dataset["validation"] = DensityDataDirSplit(
            root=self.root,
            split="validation",
            split_file=self.split_file,
        )
        self.dataset["test"] = DensityDataDirSplit(
            root=self.root,
            split="test",
            split_file=self.split_file,
        )
        self.train_set = DensityGraphDataset(self.dataset["train"], self.train_gc)
        self.val_set = DensityGraphDataset(self.dataset["validation"], self.val_gc)
        self.test_set = DensityGraphDataset(
            self.dataset["test"],
            self.test_gc,
            grid_size_file=self.grid_size_file,
            max_grid_size=self.max_grid_construction_size,
        )
        # import pdb

        # pdb.set_trace()

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self):
        loader = DataLoader(
            self.train_set,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_workers,
            collate_fn=partial(collate_list_of_dicts, pin_memory=self.pin_memory),
            # note: distributed sampler will shuffle and distribute different parts of dataset
            # to different nodes/devices
            # sampler=DistributedSampler(self.train_set, drop_last=self.drop_last),
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_set,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.val_workers,
            collate_fn=partial(collate_list_of_dicts, pin_memory=self.pin_memory),
            # note: no sampler, so all devices will get full set
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_set,
            batch_size=1,
            pin_memory=self.pin_memory,
            shuffle=False,
            num_workers=self.val_workers,
            collate_fn=partial(collate_list_of_dicts, pin_memory=self.pin_memory),
            # note: distributed sampler will shuffle and distribute different parts of dataset
            # to different nodes/devices
            # sampler=DistributedEvalSampler(self.test_set),
        )
        return loader


class DensityGraphDataset(torch.utils.data.Dataset):
    # This wrapper dataset provides a workaround such that the number
    # of probes does not need to be specified in the collate_fn of the dataloader.

    def __init__(
        self,
        dataset,
        graph_constructor,
        grid_size_file=None,
        max_grid_size=1e7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.graph_constructor = graph_constructor
        self.grid_size_file = grid_size_file
        self.max_grid_size = int(max_grid_size)

        # Infer whether splitting examples is necessary (for inference only)
        self.splitting_mode = (
            not self.grid_size_file is None
            and self.graph_constructor.num_probes is None
        )
        # Assume if num_probes is not None, then num_probes is not too large here

        # Set items in the dataset. Items in self.dataset may be repeated with different
        # probe offset values if the number of probes exceeds the self.max_grid_size
        if not self.splitting_mode:
            # Do not repeat any items in self.dataset. All probe offsets are 0
            self.dataset_indices = list(range(len(dataset)))
            self.probe_offsets = [0] * len(dataset)
            self.partial_files = [0] * len(dataset)
        else:  # splitting active
            # merge a precomputed number of probes per-material with the member list of thd dataset
            df = pd.read_csv(self.grid_size_file)
            member_list = self.dataset.dataset.data.member_list
            subset_indices = (
                self.dataset.indices
            )  # indices into member list that make up test set
            member_list_df = pd.DataFrame(
                [(i, member_list[sub], 0) for i, sub in enumerate(subset_indices)],
                columns=["dataset_index", "id", "probe_offset"],
            )
            df = df.merge(member_list_df, on="id", how="right")
            df["subset_index"] = list(range(len(df)))

            # for those materials with probe counts bigger than the construction threshold
            df["partial"] = df["Count"] > self.max_grid_size
            too_big = df[df.partial].copy()

            # get the necessary repeats to keep all items under limit
            if len(too_big) > 0:
                all_repeats = []
                for i, row in too_big.iterrows():
                    splits_dec = row["Count"] / float(self.max_grid_size)
                    splits = (
                        np.ceil(splits_dec).astype(int) - 1
                    )  # -1 for the row already present
                    offsets = [self.max_grid_size * (s + 1) for s in range(splits)]
                    repeats = df.loc[[i] * splits].assign(probe_offset=offsets)
                    all_repeats.append(repeats)

                df = pd.concat([df] + all_repeats, axis=0)

            self.dataset_indices = df["subset_index"].tolist()
            self.probe_offsets = df["probe_offset"].tolist()
            self.partial_files = df["partial"].tolist()

    def __len__(self):
        return len(self.dataset_indices)

    def __getitem__(self, index):
        start_time = time.time()

        # Get the correct file and probe offset. If too many probes, OOM during graph construction
        # can be avoided by setting lower max_grid_size.
        data_dict = self.dataset[self.dataset_indices[index]]
        probe_offset = self.probe_offsets[index]
        partial_file = self.partial_files[index]

        # 32-bit representations reduce likelihood of OOM during graph construction
        # data_dict["density"] = data_dict["density"]
        # data_dict["grid_position"] = data_dict["grid_position"]

        # Only grab probes from probe_offset
        grid_shape = torch.tensor(data_dict["density"].shape)
        if self.splitting_mode:
            # reshape density and grid position to allow for arbitrary splitting
            if (
                len(data_dict["density"].shape) == 4
            ):  # spin density TODO: have actual arg for spin
                density = data_dict["density"].reshape(-1, 1, 1, 2)[
                    probe_offset : probe_offset + self.max_grid_size
                ]
            else:
                density = data_dict["density"].reshape(-1, 1, 1)[
                    probe_offset : probe_offset + self.max_grid_size
                ]
            grid_position = data_dict["grid_position"].reshape(-1, 1, 1, 3)[
                probe_offset : probe_offset + self.max_grid_size
            ]
        else:
            density = data_dict["density"]
            grid_position = data_dict["grid_position"]

        # construct graph with probe points and atoms
        graph_dict = self.graph_constructor(
            density,
            data_dict["atoms"],
            grid_position,
        )

        graph_dict.update(
            filename=data_dict["metadata"]["filename"],
            grid_shape=grid_shape,
            probe_offset=torch.tensor(probe_offset),
            partial=torch.tensor(partial_file),
            load_time=time.time() - start_time,
        )
        return graph_dict


class DensityData(torch.utils.data.Dataset):
    def __init__(self, datapath, **kwargs):
        super().__init__(**kwargs)
        if os.path.isfile(datapath) and datapath.endswith(".tar"):
            self.data = DensityDataTar(datapath)
        elif os.path.isfile(datapath) and datapath.endswith("filelist.txt"):
            self.data = DensityPickleDir(datapath)
        elif os.path.isfile(datapath) and datapath.endswith(".txt"):
            # text file containing list of datafiles
            with open(datapath, "r") as datasetfiles:
                filelist = [
                    os.path.join(os.path.dirname(datapath), line.strip("\n"))
                    for line in datasetfiles
                ]
            self.data = ConcatDataset([DensityData(path) for path in filelist])
        elif os.path.isdir(datapath):
            self.data = DensityDataDir(datapath)
        else:
            raise ValueError("Did not find dataset at path %s", datapath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DensityDataDir(torch.utils.data.Dataset):
    def __init__(self, directory, **kwargs):
        super().__init__(**kwargs)

        self.directory = directory
        self.member_list = sorted(os.listdir(self.directory))
        self.key_to_idx = {str(k): i for i, k in enumerate(self.member_list)}

    def __len__(self):
        return len(self.member_list)

    def extractfile(self, filename):
        path = os.path.join(self.directory, filename)

        filecontent = utils.decompress_file(path)
        if path.endswith((".cube", ".cube.gz", ".cube.zz", "cube.lz4")):
            density, atoms, origin = utils.read_cube(filecontent)
        elif path.endswith(".pkl"):
            density, atoms, origin = filecontent
        else:
            density, atoms, origin = utils.read_vasp(filecontent)

        grid_pos = utils.calculate_grid_pos(density, origin, atoms.get_cell())

        metadata = {"filename": filename}
        return {
            "density": density,
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "metadata": metadata,  # Meta information
        }

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.key_to_idx[index]
        fileinfo = self.extractfile(self.member_list[index])
        # print(fileinfo["metadata"]["filename"])
        return fileinfo


class DensityDataDirSplit(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        split_file,
        extension="chgcar",
        compression=None,
        num_samples=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root = root
        self.extension = extension
        self.compression = compression
        self.file_pattern = f".{extension}"

        with open(os.path.join(split_file)) as f:
            # reverse the order so that larger molecules are tested first
            self.file_list = list(reversed(json.load(f)[split]))
        if num_samples is not None:
            self.file_list = self.file_list[:num_samples]
        self.member_list = sorted(os.listdir(self.root))
        self.member_list = [
            m for m in self.member_list if m.split(".")[0] in self.file_list
        ]
        self.key_to_idx = {str(k): i for i, k in enumerate(self.member_list)}

    def __len__(self):
        return len(self.member_list)

    def __getitem__(self, item):
        if self.compression == "lz4":
            file_name = f"{(self.file_list[item]+1):06}{self.file_pattern}"
        else:
            file_name = f"{(self.file_list[item])}{self.file_pattern}"

        try:
            # print(os.path.join(self.root, file_name))
            density, atoms, origin = utils._read_vasp2(
                os.path.join(self.root, file_name)
            )

        except EOFError:
            print("EOFError")
            print(f"Error reading {file_name} in {self.split} set, try again")
        except RuntimeError:
            print(f"Error reading {file_name} in {self.split} set")
            raise

        grid_pos = utils.calculate_grid_pos(density, origin, atoms.get_cell())

        # info = {}
        # info["file_name"] = file_name
        metadata = {"filename": file_name}

        res = {
            "density": density,
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "metadata": metadata,  # Meta information
        }
        return res


class DensityPickleDir(torch.utils.data.Dataset):
    """
    Loads density and atoms files from a directory

    Density and atoms saved separately to improve load times
    """

    def __init__(self, filename, **kwargs):
        super().__init__(**kwargs)

        if isinstance(filename, str) and Path(filename).is_file():
            with open(filename, "r") as f:
                lines = f.readlines()
            member_list = [line.replace("\n", "") for line in lines]
        else:
            raise ValueError("Need filename as input to DensityPickleDir")

        self.root = Path(filename).parent
        self.member_list = member_list

    def __len__(self):
        return len(self.member_list)

    def __getitem__(self, index):
        member = Path(self.member_list[index])
        density, atoms = utils.load_numpy_density(root=self.root, mpid=member)
        origin = np.array([0.0, 0.0, 0.0])

        grid_pos = utils.calculate_grid_pos(density, origin, atoms.get_cell())
        metadata = {"filename": str(member)}

        return {
            "density": density,
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "metadata": metadata,  # Meta information
        }


class DensityDataTar(torch.utils.data.Dataset):
    def __init__(self, tarpath, **kwargs):
        super().__init__(**kwargs)

        self.tarpath = tarpath
        self.member_list = []

        # Index tar file
        with tarfile.open(self.tarpath, "r:") as tar:
            for member in tar.getmembers():
                self.member_list.append(member)
        self.key_to_idx = {str(k): i for i, k in enumerate(self.member_list)}

    def __len__(self):
        return len(self.member_list)

    def extract_member(self, tarinfo):
        with tarfile.open(self.tarpath, "r") as tar:
            filecontent = utils.decompress_tarmember(tar, tarinfo)
            if tarinfo.name.endswith((".cube", ".cube.gz", "cube.zz", "cube.lz4")):
                density, atoms, origin = utils.read_cube(filecontent)
            else:
                density, atoms, origin = utils.read_vasp(filecontent)

        grid_pos = utils.calculate_grid_pos(density, origin, atoms.get_cell())

        metadata = {"filename": tarinfo.name}
        return {
            "density": density,
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "metadata": metadata,  # Meta information
        }

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.key_to_idx[index]
        return self.extract_member(self.member_list[index])
