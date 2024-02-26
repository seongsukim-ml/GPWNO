from typing import List, Optional
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ase.calculators.vasp import VaspChargeDensity
import ase
import asap3
import multiprocessing, math, logging, queue


Bohr = 0.529177  # Bohr radius in angstrom


def rotate_voxel(shape, cell, density, rotated_grid):
    """
    Rotate the volumetric data using trilinear interpolation.
    :param shape: voxel shape, tensor of shape (3,)
    :param cell: cell vectors, tensor of shape (3, 3)
    :param density: original density, tensor of shape (n_grid,)
    :param rotated_grid: rotated grid coordinates, tensor of shape (n_grid, 3)
    :return: rotated density, tensor of shape (n_grid,)
    """
    density = density.view(1, 1, *shape)
    rotated_grid = rotated_grid.view(1, *shape, 3)
    shape = torch.FloatTensor(shape)
    grid_cell = cell / shape.view(3, 1)
    normalized_grid = (2 * rotated_grid @ torch.linalg.inv(grid_cell) - shape + 1) / (
        shape - 1
    )
    return F.grid_sample(
        density, torch.flip(normalized_grid, [-1]), mode="bilinear", align_corners=False
    ).view(-1)


class DensityDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        split_file,
        extension="CHGCAR",
        compression="lz4",
        rotate=False,
        pbc=False,
        num_samples=None,
        reverse_order=False,
        *args,
        **kwargs,
    ):
        """
        The density dataset contains volumetric data of molecules.
        :param root: data root
        :param split: data split, can be 'train', 'validation', 'test'
        :param split_file: the data split file containing file names of the split
        :param atom_file: atom information file
        :param extension: raw data file extension, can be 'CHGCAR', 'cube', 'json'
        :param compression: raw data compression, can be 'lz4', 'xz', or None (no compression)
        :param rotate: whether to rotate the molecule and the volumetric data
        :param pbc: whether the data satisfy the periodic boundary condition
        """
        super(DensityDataset, self).__init__()
        self.root = root
        self.split = split
        self.extension = extension
        self.compression = compression
        self.rotate = rotate
        self.pbc = pbc
        self.reverse_order = reverse_order

        self.file_pattern = f".{extension}"
        if compression is not None:
            self.file_pattern += f".{compression}"
        with open(os.path.join(root, split_file)) as f:
            # reverse the order so that larger molecules are tested first
            self.file_list = list(reversed(json.load(f)[split]))
            if reverse_order:
                self.file_list = self.file_list[::-1]
            if num_samples is not None:
                self.file_list = self.file_list[:num_samples]

        if extension == "CHGCAR" or extension == "chgcar":
            pass
        else:
            raise TypeError(f"Unknown extension {extension}")

    def __getitem__(self, item):
        if self.compression == "lz4":
            file_name = f"{(self.file_list[item]+1):06}{self.file_pattern}"
        else:
            file_name = f"{(self.file_list[item])}{self.file_pattern}"

        try:
            # print(os.path.join(self.root, file_name))
            density, atoms, origin = self._read_vasp2(
                os.path.join(self.root, file_name)
            )

        except EOFError:
            print("EOFError")
            print(f"Error reading {file_name} in {self.split} set, try again")
        except RuntimeError:
            print(f"Error reading {file_name} in {self.split} set")
            raise

        grid_pos = _calculate_grid_pos(density, origin, atoms.get_cell())

        info = {}
        info["file_name"] = file_name

        res = {
            "density": density,
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
        }
        return res

    def __len__(self):
        return len(self.file_list)

    def _read_vasp(self, filepath):
        # Write to tmp file and read using ASE
        vasp_charge = VaspChargeDensity(filename=filepath)
        density = vasp_charge.chg[-1]  # separate density
        atoms = vasp_charge.atoms[-1]  # separate atom positions

        return (
            density,
            atoms,
            np.zeros(3),
        )  # TODO: Can we always assume origin at 0,0,0?

    def _read_vasp2(self, filepath):
        fileobj = open(filepath, "r")
        readline = fileobj.readline
        readline()  # the first comment line
        scale = float(readline())  # the scaling factor (lattice constant)

        # the upcoming three lines contain the cell information
        cell = torch.empty(3, 3, dtype=torch.float)
        for i in range(3):
            cell[i] = torch.FloatTensor([float(s) for s in readline().split()])
        cell = cell * scale

        # the sixth line specifies the constituting elements
        elements = readline().split()
        # the seventh line supplies the number of atoms per atomic species
        n_atoms = [int(s) for s in readline().split()]
        # the eighth line is always "Direct" in our application
        readline()

        tot_atoms = sum(n_atoms)
        atom_type = []
        atom_coord = torch.empty(tot_atoms, 3, dtype=torch.float)
        # the upcoming lines contains the atomic positions in fractional coordinates
        idx = 0
        for elem, n in zip(elements, n_atoms):
            atom_type += [elem] * n
            for _ in range(n):
                atom_coord[idx] = torch.FloatTensor(
                    [float(s) for s in readline().split()]
                )
                idx += 1

        # the coordinates are fractional, convert them to cartesian
        atom_coord = atom_coord @ cell
        g = Data(x=atom_type, pos=atom_coord)

        readline()  # an empty line
        shape = [int(s) for s in readline().split()]  # grid size
        n_grid = shape[0] * shape[1] * shape[2]
        # the grids are corner-aligned
        x_coord = (
            torch.linspace(0, shape[0] - 1, shape[0]).unsqueeze(-1) / shape[0] * cell[0]
        )
        y_coord = (
            torch.linspace(0, shape[1] - 1, shape[1]).unsqueeze(-1) / shape[1] * cell[1]
        )
        z_coord = (
            torch.linspace(0, shape[2] - 1, shape[2]).unsqueeze(-1) / shape[2] * cell[2]
        )
        grid_coord = (
            x_coord.view(-1, 1, 1, 3)
            + y_coord.view(1, -1, 1, 3)
            + z_coord.view(1, 1, -1, 3)
        )
        grid_coord = grid_coord.view(-1, 3)

        # the augmented occupancies are ignored
        density = torch.FloatTensor([float(s) for s in fileobj.read().split()[:n_grid]])
        # the value stored is the charge within a grid instead of the charge density
        # divide the charge by the grid volume to get the density
        volume = torch.linalg.det(cell).abs()
        density = density / volume
        # CHGCAR file stores the density as Z-Y-X, convert them to X-Y-Z
        density = (
            density.view(shape[2], shape[1], shape[0]).transpose(0, 2).contiguous()
        )
        atoms = ase.Atoms(symbols=atom_type, positions=atom_coord, cell=cell.numpy())
        return density, atoms, np.zeros(3)


def pad_and_stack(tensors: List[torch.Tensor]):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0
        )
    return torch.stack(tensors)


def _cell_heights(cell_object):
    volume = cell_object.volume
    crossproducts = np.cross(cell_object[[1, 2, 0]], cell_object[[2, 0, 1]])
    crosslengths = np.sqrt(np.sum(np.square(crossproducts), axis=1))
    heights = volume / crosslengths
    return heights


def _calculate_grid_pos(density, origin, cell):
    # Calculate grid positions
    ngridpts = np.array(density.shape)  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    return grid_pos


def atoms_and_probe_sample_to_graph_dict(density, atoms, grid_pos, cutoff, num_probes):
    # Sample probes on the calculated grid
    probe_choice_max = np.prod(grid_pos.shape[0:3])
    probe_choice = np.random.randint(probe_choice_max, size=num_probes)
    probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
    probe_pos = grid_pos[probe_choice]
    probe_target = density[probe_choice]

    atom_edges, atom_edges_displacement, neighborlist, inv_cell_T = atoms_to_graph(
        atoms, cutoff
    )
    probe_edges, probe_edges_displacement = probes_to_graph(
        atoms, probe_pos, cutoff, neighborlist=neighborlist, inv_cell_T=inv_cell_T
    )

    default_type = torch.get_default_dtype()

    if not probe_edges:
        probe_edges = [np.zeros((0, 2), dtype=np.int)]
        probe_edges_displacement = [np.zeros((0, 3), dtype=np.int)]
    # pylint: disable=E1102
    res = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": torch.tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
        "probe_edges": torch.tensor(np.concatenate(probe_edges, axis=0)),
        "probe_edges_displacement": torch.tensor(
            np.concatenate(probe_edges_displacement, axis=0), dtype=default_type
        ),
        "probe_target": probe_target,
    }
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["num_probe_edges"] = torch.tensor(res["probe_edges"].shape[0])
    res["num_probes"] = torch.tensor(res["probe_target"].shape[0])
    res["probe_xyz"] = torch.tensor(probe_pos, dtype=default_type)
    res["atom_xyz"] = torch.tensor(atoms.get_positions(), dtype=default_type)
    res["cell"] = torch.tensor(np.array(atoms.get_cell()), dtype=default_type)

    return res


def atoms_to_graph_dict(atoms, cutoff):
    atom_edges, atom_edges_displacement, _, _ = atoms_to_graph(atoms, cutoff)

    default_type = torch.get_default_dtype()

    # pylint: disable=E1102
    res = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": torch.tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
    }
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["atom_xyz"] = torch.tensor(atoms.get_positions(), dtype=default_type)
    res["cell"] = torch.tensor(np.array(atoms.get_cell()), dtype=default_type)

    return res


class AseNeigborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist

    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
            self.atoms_positions[indices]
            + offsets @ self.atoms_cell
            - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)

        return indices, rel_positions, dist2


def atoms_to_graph(atoms, cutoff):
    atom_edges = []
    atom_edges_displacement = []

    inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    # Compute neighborlist
    if np.any(atoms.get_cell().lengths() <= 0.0001) or (
        np.any(atoms.get_pbc()) and np.any(_cell_heights(atoms.get_cell()) < cutoff)
    ):
        neighborlist = AseNeigborListWrapper(cutoff, atoms)
    else:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)

    atom_positions = atoms.get_positions()

    for i in range(len(atoms)):
        neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, cutoff)

        self_index = np.ones_like(neigh_idx) * i
        edges = np.stack((neigh_idx, self_index), axis=1)

        neigh_pos = atom_positions[neigh_idx]
        this_pos = atom_positions[i]
        neigh_origin = neigh_vec + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        atom_edges.append(edges)
        atom_edges_displacement.append(neigh_origin_scaled)

    return atom_edges, atom_edges_displacement, neighborlist, inv_cell_T


def probes_to_graph(atoms, probe_pos, cutoff, neighborlist=None, inv_cell_T=None):
    # pdb.set_trace()
    probe_edges = []
    probe_edges_displacement = []
    if inv_cell_T is None:
        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    if hasattr(neighborlist, "get_neighbors_querypoint"):
        results = neighborlist.get_neighbors_querypoint(probe_pos, cutoff)
        atomic_numbers = atoms.get_atomic_numbers()
    else:
        # Insert probe atoms
        num_probes = probe_pos.shape[0]
        probe_atoms = ase.Atoms(numbers=[0] * num_probes, positions=probe_pos)
        atoms_with_probes = atoms.copy()
        atoms_with_probes.extend(probe_atoms)
        atomic_numbers = atoms_with_probes.get_atomic_numbers()

        if np.any(atoms.get_cell().lengths() <= 0.0001) or (
            np.any(atoms.get_pbc()) and np.any(_cell_heights(atoms.get_cell()) < cutoff)
        ):
            neighborlist = AseNeigborListWrapper(cutoff, atoms_with_probes)
        else:
            neighborlist = asap3.FullNeighborList(cutoff, atoms_with_probes)

        results = [
            neighborlist.get_neighbors(i + len(atoms), cutoff)
            for i in range(num_probes)
        ]

    atom_positions = atoms.get_positions()
    for i, (neigh_idx, neigh_vec, _) in enumerate(results):
        neigh_atomic_species = atomic_numbers[neigh_idx]

        neigh_is_atom = neigh_atomic_species != 0
        neigh_atoms = neigh_idx[neigh_is_atom]
        self_index = np.ones_like(neigh_atoms) * i
        edges = np.stack((neigh_atoms, self_index), axis=1)

        neigh_pos = atom_positions[neigh_atoms]
        this_pos = probe_pos[i]
        neigh_origin = neigh_vec[neigh_is_atom] + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        probe_edges.append(edges)
        probe_edges_displacement.append(neigh_origin_scaled)

    return probe_edges, probe_edges_displacement


def collate_list_of_dicts(list_of_dicts, pin_memory=False):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

    # Convert each list of tensors to single tensor with pad and stack
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x

    collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
    return collated


class CollateFuncRandomSample:
    def __init__(self, cutoff, n_samples, pin_memory=False, set_pbc_to=None):
        self.num_probes = n_samples
        self.cutoff = cutoff
        self.pin_memory = pin_memory
        self.set_pbc = set_pbc_to

    def __call__(self, input_dicts: List):
        graphs = []
        for i in input_dicts:
            if self.set_pbc is not None:
                atoms = i["atoms"].copy()
                atoms.set_pbc(self.set_pbc)
            else:
                atoms = i["atoms"]
            if self.num_probes is None:
                return input_dicts[0]
            else:
                graphs.append(
                    atoms_and_probe_sample_to_graph_dict(
                        i["density"],
                        atoms,
                        i["grid_position"],
                        self.cutoff,
                        self.num_probes,
                    )
                )

        return collate_list_of_dicts(graphs, pin_memory=self.pin_memory)


class CollateFuncAtoms:
    def __init__(self, cutoff, n_samples, pin_memory=False, set_pbc_to=None):
        self.cutoff = cutoff
        self.pin_memory = pin_memory
        self.set_pbc = set_pbc_to

    def __call__(self, input_dicts: List):
        graphs = []
        for i in input_dicts:
            if self.set_pbc is not None:
                atoms = i["atoms"].copy()
                atoms.set_pbc(self.set_pbc)
            else:
                atoms = i["atoms"]

            graphs.append(
                atoms_to_graph_dict(
                    atoms,
                    self.cutoff,
                )
            )

        return collate_list_of_dicts(graphs, pin_memory=self.pin_memory)


def grid_iterator_worker(
    atoms, meshgrid, probe_count, cutoff, slice_id_queue, result_queue
):
    try:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)
    except Exception as e:
        logging.info(
            "Failed to create asap3 neighborlist, this might be very slow. Error: %s", e
        )
        neighborlist = None
    while True:
        try:
            slice_id = slice_id_queue.get(True, 1)
        except queue.Empty:
            while not result_queue.empty():
                time.sleep(1)
            result_queue.close()
            return 0
        res = DensityGridIterator.static_get_slice(
            slice_id, atoms, meshgrid, probe_count, cutoff, neighborlist=neighborlist
        )
        result_queue.put((slice_id, res))


class DensityGridIterator:
    def __init__(
        self,
        densitydict,
        probe_count: int,
        cutoff: float,
        set_pbc_to: Optional[bool] = None,
    ):
        num_positions = np.prod(densitydict["grid_position"].shape[0:3])
        self.num_slices = int(math.ceil(num_positions / probe_count))
        self.probe_count = probe_count
        self.cutoff = cutoff
        self.set_pbc = set_pbc_to

        if self.set_pbc is not None:
            self.atoms = densitydict["atoms"].copy()
            self.atoms.set_pbc(self.set_pbc)
        else:
            self.atoms = densitydict["atoms"]

        self.meshgrid = densitydict["grid_position"]

    def get_slice(self, slice_index):
        return self.static_get_slice(
            slice_index, self.atoms, self.meshgrid, self.probe_count, self.cutoff
        )

    @staticmethod
    def static_get_slice(
        slice_index, atoms, meshgrid, probe_count, cutoff, neighborlist=None
    ):
        num_positions = np.prod(meshgrid.shape[0:3])
        flat_index = np.arange(
            slice_index * probe_count,
            min((slice_index + 1) * probe_count, num_positions),
        )
        pos_index = np.unravel_index(flat_index, meshgrid.shape[0:3])
        probe_pos = meshgrid[pos_index]
        probe_edges, probe_edges_displacement = probes_to_graph(
            atoms, probe_pos, cutoff, neighborlist
        )

        if not probe_edges:
            probe_edges = [np.zeros((0, 2), dtype=np.int)]
            probe_edges_displacement = [np.zeros((0, 3), dtype=np.float32)]

        res = {
            "probe_edges": np.concatenate(probe_edges, axis=0),
            "probe_edges_displacement": np.concatenate(
                probe_edges_displacement, axis=0
            ).astype(np.float32),
        }
        res["num_probe_edges"] = res["probe_edges"].shape[0]
        res["num_probes"] = len(flat_index)
        res["probe_xyz"] = probe_pos.astype(np.float32)

        return res

    def __iter__(self):
        self.current_slice = 0
        slice_id_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue(100)
        self.finished_slices = dict()
        for i in range(self.num_slices):
            slice_id_queue.put(i)
        self.workers = [
            multiprocessing.Process(
                target=grid_iterator_worker,
                args=(
                    self.atoms,
                    self.meshgrid,
                    self.probe_count,
                    self.cutoff,
                    slice_id_queue,
                    self.result_queue,
                ),
            )
            for _ in range(6)
        ]
        for w in self.workers:
            w.start()
        return self

    def __next__(self):
        if self.current_slice < self.num_slices:
            this_slice = self.current_slice
            self.current_slice += 1

            # Retrieve finished slices until we get the one we are looking for
            while this_slice not in self.finished_slices:
                i, res = self.result_queue.get()
                res = {
                    k: torch.tensor(v) for k, v in res.items()
                }  # convert to torch tensor
                self.finished_slices[i] = res
            return self.finished_slices.pop(this_slice)
        else:
            for w in self.workers:
                w.join()
            raise StopIteration
