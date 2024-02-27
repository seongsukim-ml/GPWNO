import itertools
import logging

from contextlib import contextmanager
from typing import Callable, Optional, TypedDict, Union
from collections import namedtuple

import math
import hydra
import numpy as np
from scipy.special import binom

import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch_geometric.nn.models.schnet import GaussianSmearing


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        # hparams 저장하고 hparams으로 불러 올 수 있음

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return {"optimizer": opt}
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "frequency": 1,
            "strict": False,
            "monitor": "val/loss",
        }


def define_hyperparams(dictionary):
    global ModuleHyperparams
    ModuleHyperparams = namedtuple("ModuleHyperparams", dictionary)
    out_object = ModuleHyperparams(**dictionary)

    return out_object


class BroadcastGTOTensor(nn.Module):
    r"""
    Broadcast between spherical tensors of the Gaussian Type Orbitals (GTOs):

    .. math::
        \{a_{clm}, 1\le c\le c_{max}, 0\le\ell\le\ell_{max}, -\ell\le m\le\ell\}

    For efficiency reason, the feature tensor is indexed by l, c, m.
    For example, for lmax = 3, cmax = 2, we have a tensor of 1s2s 1p2p 1d2d 1f2f.
    Currently, we support the following broadcasting:
    lc -> lcm;
    m -> lcm.
    """

    def __init__(self, lmax, cmax, src="lc", dst="lcm"):
        super(BroadcastGTOTensor, self).__init__()
        assert src in ["lc", "m"]
        assert dst in ["lcm"]
        self.src = src
        self.dst = dst
        self.lmax = lmax
        self.cmax = cmax

        if src == "lc":
            self.src_dim = (lmax + 1) * cmax
        else:
            self.src_dim = (lmax + 1) ** 2
        self.dst_dim = (lmax + 1) ** 2 * cmax

        if src == "lc":
            indices = self._generate_lc2lcm_indices()
        else:
            indices = self._generate_m2lcm_indices()
        self.register_buffer("indices", indices)

    def _generate_lc2lcm_indices(self):
        r"""
        lc -> lcm
        .. math::
            1s2s 1p2p → 1s2s 1p_x1p_y1p_z2p_x2p_y2p_z
        [0, 1, 2, 2, 2, 3, 3, 3]

        :return: (lmax+1)^2 * cmax
        """
        indices = [
            l * self.cmax + c
            for l in range(self.lmax + 1)
            for c in range(self.cmax)
            for _ in range(2 * l + 1)
        ]
        return torch.LongTensor(indices)

    def _generate_m2lcm_indices(self):
        r"""
        m -> lcm
        .. math::
            s p_x p_y p_z → 1s2s 1p_x1p_y1p_z2p_x2p_y2p_z
        [0, 0, 1, 2, 3, 1, 2, 3]

        :return: (lmax+1)^2 * cmax
        """
        indices = [
            l * l + m
            for l in range(self.lmax + 1)
            for _ in range(self.cmax)
            for m in range(2 * l + 1)
        ]
        return torch.LongTensor(indices)

    def forward(self, x):
        """
        Apply broadcasting to x.
        :param x: (..., src_dim)
        :return: (..., dst_dim)
        """
        assert x.size(-1) == self.src_dim, (
            f"Input dimension mismatch! "
            f"Should be {self.src_dim}, but got {x.size(-1)} instead!"
        )
        if self.src == self.dst:
            return x
        return x[..., self.indices]


class _Stats(TypedDict):
    variance_in: float
    variance_out: float
    n_samples: int


IndexFn = Callable[[], None]


def _check_consistency(old: torch.Tensor, new: torch.Tensor, key: str):
    if not torch.allclose(old, new):
        raise ValueError(
            f"Scale factor parameter {key} is inconsistent with the loaded state dict.\n"
            f"Old: {old}\n"
            f"Actual: {new}"
        )


class ScaleFactor(nn.Module):
    scale_factor: torch.Tensor

    name: Optional[str] = None
    index_fn: Optional[IndexFn] = None
    stats: Optional[_Stats] = None

    def __init__(
        self,
        name: Optional[str] = None,
        enforce_consistency: bool = True,
    ):
        super().__init__()

        self.name = name
        self.index_fn = None
        self.stats = None

        self.scale_factor = nn.parameter.Parameter(
            torch.tensor(0.0), requires_grad=False
        )
        if enforce_consistency:
            self._register_load_state_dict_pre_hook(self._enforce_consistency)

    def _enforce_consistency(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        if not self.fitted:
            return

        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key not in state_dict:
                continue

            input_param = state_dict[key]
            _check_consistency(old=param, new=input_param, key=key)

    @property
    def fitted(self):
        return bool((self.scale_factor != 0.0).item())

    @torch.jit.unused
    def reset_(self):
        self.scale_factor.zero_()

    @torch.jit.unused
    def set_(self, scale: Union[float, torch.Tensor]):
        if self.fitted:
            _check_consistency(
                old=self.scale_factor,
                new=torch.tensor(scale) if isinstance(scale, float) else scale,
                key="scale_factor",
            )
        self.scale_factor.fill_(scale)

    @torch.jit.unused
    def initialize_(self, *, index_fn: Optional[IndexFn] = None):
        self.index_fn = index_fn

    @contextmanager
    @torch.jit.unused
    def fit_context_(self):
        self.stats = _Stats(variance_in=0.0, variance_out=0.0, n_samples=0)
        yield
        del self.stats
        self.stats = None

    @torch.jit.unused
    def fit_(self):
        assert self.stats, "Stats not set"
        for k, v in self.stats.items():
            assert v > 0, f"{k} is {v}"

        self.stats["variance_in"] = self.stats["variance_in"] / self.stats["n_samples"]
        self.stats["variance_out"] = (
            self.stats["variance_out"] / self.stats["n_samples"]
        )

        ratio = self.stats["variance_out"] / self.stats["variance_in"]
        value = math.sqrt(1 / ratio)

        self.set_(value)

        stats = dict(**self.stats)
        return stats, ratio, value

    @torch.no_grad()
    @torch.jit.unused
    def _observe(self, x: torch.Tensor, ref: Optional[torch.Tensor] = None):
        if self.stats is None:
            logging.debug("Observer not initialized but self.observe() called")
            return

        n_samples = x.shape[0]
        self.stats["variance_out"] += torch.mean(torch.var(x, dim=0)).item() * n_samples

        if ref is None:
            self.stats["variance_in"] += n_samples
        else:
            self.stats["variance_in"] += (
                torch.mean(torch.var(ref, dim=0)).item() * n_samples
            )
        self.stats["n_samples"] += n_samples

    def forward(
        self,
        x: torch.Tensor,
        *,
        ref: Optional[torch.Tensor] = None,
    ):
        if self.index_fn is not None:
            self.index_fn()

        if self.fitted:
            x = x * self.scale_factor

        if not torch.jit.is_scripting():
            self._observe(x, ref=ref)

        return x


def _standardize(kernel):
    """
    Makes sure that N*Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He (Kaiming) initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor


class Dense(torch.nn.Module):
    """
    Combines dense layer with scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(self, in_features, out_features, bias=False, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

    def reset_parameters(self, initializer=he_orthogonal_init):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = self._activation(x)
        return x


class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return x * self._activation(x)


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        layer_kwargs: str
            Keyword arguments for initializing the layers.
    """

    def __init__(self, units: int, nLayers: int = 2, layer=Dense, **layer_kwargs):
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                layer(in_features=units, out_features=units, bias=False, **layer_kwargs)
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x


class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        exponent: int
            Exponent of the envelope function.
    """

    def __init__(self, exponent):
        super().__init__()
        assert exponent > 0
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class ExponentialEnvelope(torch.nn.Module):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """

    def __init__(self):
        super().__init__()

    def forward(self, d_scaled):
        env_val = torch.exp(-(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled)))
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class SphericalBesselBasis(torch.nn.Module):
    """
    1D spherical Bessel basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
    ):
        super().__init__()
        self.norm_const = math.sqrt(2 / (cutoff**3))
        # cutoff ** 3 to counteract dividing by d_scaled = d / cutoff

        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=torch.tensor(np.pi * np.arange(1, num_radial + 1, dtype=np.float32)),
            requires_grad=True,
        )

    def forward(self, d_scaled):
        return (
            self.norm_const
            / d_scaled[:, None]
            * torch.sin(self.frequencies * d_scaled[:, None])
        )  # (num_edges, num_radial)


class BernsteinBasis(torch.nn.Module):
    """
    Bernstein polynomial basis,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    pregamma_initial: float
        Initial value of exponential coefficient gamma.
        Default: gamma = 0.5 * a_0**-1 = 0.94486,
        inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
    """

    def __init__(
        self,
        num_radial: int,
        pregamma_initial: float = 0.45264,
    ):
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer(
            "prefactor",
            torch.tensor(prefactor, dtype=torch.float),
            persistent=False,
        )

        self.pregamma = torch.nn.Parameter(
            data=torch.tensor(pregamma_initial, dtype=torch.float),
            requires_grad=True,
        )
        self.softplus = torch.nn.Softplus()

        exp1 = torch.arange(num_radial)
        self.register_buffer("exp1", exp1[None, :], persistent=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer("exp2", exp2[None, :], persistent=False)

    def forward(self, d_scaled):
        gamma = self.softplus(self.pregamma)  # constrain to positive
        exp_d = torch.exp(-gamma * d_scaled)[:, None]
        return self.prefactor * (exp_d**self.exp1) * ((1 - exp_d) ** self.exp2)


class RadialBasis(torch.nn.Module):
    """

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
    ):
        super().__init__()
        self.inv_cutoff = 1 / cutoff

        env_name = envelope["name"].lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]

        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]

        # RBFs get distances scaled to be in [0, 1]
        if rbf_name == "gaussian":
            self.rbf = GaussianSmearing(
                start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            )
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(
                num_radial=num_radial, cutoff=cutoff, **rbf_hparams
            )
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d):
        d_scaled = d * self.inv_cutoff

        env = self.envelope(d_scaled)
        return env[:, None] * self.rbf(d_scaled)  # (nEdges, num_radial)


import torch
import numpy as np
import scipy.io

# import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter

#################################################
#
# Utilities
#
#################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class MatReader(object):
#     def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
#         super(MatReader, self).__init__()

#         self.to_torch = to_torch
#         self.to_cuda = to_cuda
#         self.to_float = to_float

#         self.file_path = file_path

#         self.data = None
#         self.old_mat = None
#         self._load_file()

#     def _load_file(self):
#         try:
#             self.data = scipy.io.loadmat(self.file_path)
#             self.old_mat = True
#         except:
#             self.data = h5py.File(self.file_path)
#             self.old_mat = False

#     def load_file(self, file_path):
#         self.file_path = file_path
#         self._load_file()

#     def read_field(self, field):
#         x = self.data[field]

#         if not self.old_mat:
#             x = x[()]
#             x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

#         if self.to_float:
#             x = x.astype(np.float32)

#         if self.to_torch:
#             x = torch.from_numpy(x)

#             if self.to_cuda:
#                 x = x.cuda()

#         return x

#     def set_cuda(self, to_cuda):
#         self.to_cuda = to_cuda

#     def set_torch(self, to_torch):
#         self.to_torch = to_torch

#     def set_float(self, to_float):
#         self.to_float = to_float


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0).view(-1)
        self.std = torch.std(x, 0).view(-1)

        self.eps = eps

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.view(s)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            std = self.std[sample_idx] + self.eps  # batch * n
            mean = self.mean[sample_idx]

        s = x.size()
        x = x.view(s[0], -1)
        x = (x * std) + mean
        x = x.view(s)
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


class SquareMeshGenerator(object):
    def __init__(self, real_space, mesh_size):
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape(
                (self.n, 1)
            )
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(
                    np.linspace(real_space[j][0], real_space[j][1], mesh_size[j])
                )
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-(pwd**2) / sigma**2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0 : 2 * self.d] = self.grid[self.edge_index.T].reshape(
                    (self.n_edges, -1)
                )
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0 : self.d], xy[:, self.d :])
            else:
                edge_attr = f(
                    xy[:, 0 : self.d],
                    xy[:, self.d :],
                    theta[self.edge_index[0]],
                    theta[self.edge_index[1]],
                )

        return torch.tensor(edge_attr, dtype=torch.float)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return torch.tensor(self.edge_index_boundary, dtype=torch.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape(
                    (self.n_edges_boundary, -1)
                )
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3 * self.d))
                edge_attr_boundary[:, 0 : 2 * self.d] = self.grid[
                    self.edge_index_boundary.T
                ].reshape((self.n_edges_boundary, -1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d + 1] = theta[
                    self.edge_index_boundary[1]
                ]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape(
                (self.n_edges_boundary, -1)
            )
            if theta is None:
                edge_attr_boundary = f(xy[:, 0 : self.d], xy[:, self.d :])
            else:
                edge_attr_boundary = f(
                    xy[:, 0 : self.d],
                    xy[:, self.d :],
                    theta[self.edge_index_boundary[0]],
                    theta[self.edge_index_boundary[1]],
                )

        return torch.tensor(edge_attr_boundary, dtype=torch.float)


class RandomMeshGenerator(object):
    def __init__(self, real_space, mesh_size, sample_size):
        super(RandomMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_size

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape(
                (self.n, 1)
            )
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(
                    np.linspace(real_space[j][0], real_space[j][1], mesh_size[j])
                )
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        if self.m > self.n:
            self.m = self.n

        self.idx = np.array(range(self.n))
        self.grid_sample = self.grid

    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[: self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        rbf = np.exp(-(pwd**2) / sigma**2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                theta = theta[self.idx]
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0 : 2 * self.d] = self.grid_sample[
                    self.edge_index.T
                ].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0 : self.d], xy[:, self.d :])
            else:
                theta = theta[self.idx]
                edge_attr = f(
                    xy[:, 0 : self.d],
                    xy[:, self.d :],
                    theta[self.edge_index[0]],
                    theta[self.edge_index[1]],
                )

        return torch.tensor(edge_attr, dtype=torch.float)

    # def get_boundary(self):
    #     s = self.s
    #     n = self.n
    #     boundary1 = np.array(range(0, s))
    #     boundary2 = np.array(range(n - s, n))
    #     boundary3 = np.array(range(s, n, s))
    #     boundary4 = np.array(range(2 * s - 1, n, s))
    #     self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])
    #
    # def boundary_connectivity2d(self, stride=1):
    #
    #     boundary = self.boundary[::stride]
    #     boundary_size = len(boundary)
    #     vertice1 = np.array(range(self.n))
    #     vertice1 = np.repeat(vertice1, boundary_size)
    #     vertice2 = np.tile(boundary, self.n)
    #     self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
    #     self.n_edges_boundary = self.edge_index_boundary.shape[1]
    #     return torch.tensor(self.edge_index_boundary, dtype=torch.long)
    #
    # def attributes_boundary(self, f=None, theta=None):
    #     # if self.edge_index_boundary == None:
    #     #     self.boundary_connectivity2d()
    #     if f is None:
    #         if theta is None:
    #             edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #         else:
    #             edge_attr_boundary = np.zeros((self.n_edges_boundary, 3*self.d))
    #             edge_attr_boundary[:,0:2*self.d] = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #             edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
    #             edge_attr_boundary[:, 2 * self.d +1] = theta[self.edge_index_boundary[1]]
    #     else:
    #         xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #         if theta is None:
    #             edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:])
    #         else:
    #             edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index_boundary[0]], theta[self.edge_index_boundary[1]])
    #
    #     return torch.tensor(edge_attr_boundary, dtype=torch.float)


class RandomGridSplitter(object):
    def __init__(self, grid, resolution, m=200, l=2, radius=0.25):
        super(RandomGridSplitter, self).__init__()

        self.grid = grid
        self.resolution = resolution
        self.n = resolution**2
        self.m = m
        self.l = l
        self.radius = radius

        assert self.n % self.m == 0
        self.num = self.n // self.m

    def get_data(self, theta):

        data = []
        for i in range(self.l):
            perm = torch.randperm(self.n)
            perm = perm.reshape(self.num, self.m)

            for j in range(self.num):
                idx = perm[j, :].reshape(
                    -1,
                )
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                X = torch.cat([grid_sample, theta_sample], dim=1)

                pwd = sklearn.metrics.pairwise_distances(grid_sample)
                edge_index = np.vstack(np.where(pwd <= self.radius))
                n_edges = edge_index.shape[1]
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                edge_attr = np.zeros((n_edges, 6))
                a = theta_sample[:, 0]
                edge_attr[:, :4] = grid_sample[edge_index.T].reshape(n_edges, -1)
                edge_attr[:, 4] = a[edge_index[0]]
                edge_attr[:, 5] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                data.append(
                    Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=idx)
                )
        print("test", len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.num * self.l // batch_size2

        out = torch.zeros(
            self.n,
        )
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i].reshape(batch_size2, self.m)
            for j in range(batch_size2):
                pred_ij = pred_i[j, :].reshape(
                    -1,
                )
                idx = split_idx_i[j, :].reshape(
                    -1,
                )
                out[idx] = pred_ij

        out = out / self.l

        # out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        # out = torch.tensor(out, dtype=torch.float)
        return out.reshape(
            -1,
        )


class DownsampleGridSplitter(object):
    def __init__(self, grid, resolution, r, m=100, radius=0.15, edge_features=1):
        super(DownsampleGridSplitter, self).__init__()

        self.grid = grid.reshape(resolution, resolution, 2)
        # self.theta = theta.reshape(resolution, resolution,-1)
        # self.y = y.reshape(resolution, resolution,1)
        self.resolution = resolution
        if resolution % 2 == 1:
            self.s = int(((resolution - 1) / r) + 1)
        else:
            self.s = int(resolution / r)
        self.r = r
        self.n = resolution**2
        self.m = m
        self.radius = radius
        self.edge_features = edge_features

        self.index = torch.tensor(range(self.n), dtype=torch.long).reshape(
            self.resolution, self.resolution
        )

    def ball_connectivity(self, grid):
        pwd = sklearn.metrics.pairwise_distances(grid)
        edge_index = np.vstack(np.where(pwd <= self.radius))
        n_edges = edge_index.shape[1]
        return torch.tensor(edge_index, dtype=torch.long), n_edges

    def get_data(self, theta):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        data = []
        for x in range(self.r):
            for y in range(self.r):
                grid_sub = self.grid[x :: self.r, y :: self.r, :].reshape(-1, 2)
                theta_sub = theta[x :: self.r, y :: self.r, :].reshape(-1, theta_d)

                perm = torch.randperm(self.n)
                m = self.m - grid_sub.shape[0]
                idx = perm[:m]
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                grid_split = torch.cat([grid_sub, grid_sample], dim=0)
                theta_split = torch.cat([theta_sub, theta_sample], dim=0)
                X = torch.cat([grid_split, theta_split], dim=1)

                edge_index, n_edges = self.ball_connectivity(grid_split)

                edge_attr = np.zeros((n_edges, 4 + self.edge_features * 2))
                a = theta_split[:, : self.edge_features]
                edge_attr[:, :4] = grid_split[edge_index.T].reshape(n_edges, -1)
                edge_attr[:, 4 : 4 + self.edge_features] = a[edge_index[0]]
                edge_attr[:, 4 + self.edge_features : 4 + self.edge_features * 2] = a[
                    edge_index[1]
                ]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)

                data.append(
                    Data(
                        x=X,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        split_idx=split_idx,
                    )
                )
        print("test", len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def sample(self, theta, Y):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.resolution, self.resolution)

        x = torch.randint(0, self.r, (1,))
        y = torch.randint(0, self.r, (1,))

        grid_sub = self.grid[x :: self.r, y :: self.r, :].reshape(-1, 2)
        theta_sub = theta[x :: self.r, y :: self.r, :].reshape(-1, theta_d)
        Y_sub = Y[x :: self.r, y :: self.r].reshape(
            -1,
        )
        index_sub = self.index[x :: self.r, y :: self.r].reshape(
            -1,
        )
        n_sub = Y_sub.shape[0]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(
                self.n,
            )[idx]

            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=0).reshape(
                -1,
            )
            index_split = torch.cat([index_sub, idx], dim=0).reshape(
                -1,
            )
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(
                -1,
            )
            index_split = index_sub.reshape(
                -1,
            )
            X = torch.cat([grid_split, theta_split], dim=1)

        edge_index, n_edges = self.ball_connectivity(grid_split)

        edge_attr = np.zeros((n_edges, 4 + self.edge_features * 2))
        a = theta_split[:, : self.edge_features]
        edge_attr[:, :4] = grid_split[edge_index.T].reshape(n_edges, -1)
        edge_attr[:, 4 : 4 + self.edge_features] = a[edge_index[0]]
        edge_attr[:, 4 + self.edge_features : 4 + self.edge_features * 2] = a[
            edge_index[1]
        ]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        data = Data(
            x=X,
            y=Y_split,
            edge_index=edge_index,
            edge_attr=edge_attr,
            split_idx=split_idx,
            sample_idx=index_split,
        )
        print(
            "train",
            X.shape,
            Y_split.shape,
            edge_index.shape,
            edge_attr.shape,
            index_split.shape,
        )

        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.r**2 // batch_size2

        out = torch.zeros((self.resolution, self.resolution))
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i]
            for j in range(batch_size2):
                pred_ij = pred_i[j, :]
                x, y = split_idx_i[j]
                if self.resolution % 2 == 1:
                    if x == 0:
                        nx = self.s
                    else:
                        nx = self.s - 1
                    if y == 0:
                        ny = self.s
                    else:
                        ny = self.s - 1
                else:
                    nx = self.s
                    ny = self.s
                # pred_ij = pred_i[idx : idx + nx * ny]
                out[x :: self.r, y :: self.r] = pred_ij[: nx * ny].reshape(nx, ny)

        out = gaussian_filter(out, sigma=sigma, mode="constant", cval=0)
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(
            -1,
        )


class TorusGridSplitter(object):
    def __init__(self, grid, resolution, r, m=100, radius=0.15, edge_features=1):
        super(TorusGridSplitter, self).__init__()

        self.grid = grid.reshape(resolution, resolution, 2)
        # self.theta = theta.reshape(resolution, resolution,-1)
        # self.y = y.reshape(resolution, resolution,1)
        self.resolution = resolution
        if resolution % 2 == 1:
            self.s = int(((resolution - 1) / r) + 1)
        else:
            self.s = int(resolution / r)
        self.r = r
        self.n = resolution**2
        self.m = m
        self.radius = radius
        self.edge_features = edge_features

        self.index = torch.tensor(range(self.n), dtype=torch.long).reshape(
            self.resolution, self.resolution
        )

    def pairwise_difference(self, grid1, grid2):
        n = grid1.shape[0]
        x1 = grid1[:, 0]
        y1 = grid1[:, 1]
        x2 = grid2[:, 0]
        y2 = grid2[:, 1]

        X1 = np.tile(x1.reshape(n, 1), [1, n])
        X2 = np.tile(x2.reshape(1, n), [n, 1])
        X_diff = X1 - X2
        Y1 = np.tile(y1.reshape(n, 1), [1, n])
        Y2 = np.tile(y2.reshape(1, n), [n, 1])
        Y_diff = Y1 - Y2

        return X_diff, Y_diff

    def torus_connectivity(self, grid):
        pwd0 = sklearn.metrics.pairwise_distances(grid, grid)
        X_diff0, Y_diff0 = self.pairwise_difference(grid, grid)

        grid1 = grid
        grid1[:, 0] = grid[:, 0] + 1
        pwd1 = sklearn.metrics.pairwise_distances(grid, grid1)
        X_diff1, Y_diff1 = self.pairwise_difference(grid, grid1)

        grid2 = grid
        grid2[:, 1] = grid[:, 1] + 1
        pwd2 = sklearn.metrics.pairwise_distances(grid, grid2)
        X_diff2, Y_diff2 = self.pairwise_difference(grid, grid2)

        grid3 = grid
        grid3[:, :] = grid[:, :] + 1
        pwd3 = sklearn.metrics.pairwise_distances(grid, grid3)
        X_diff3, Y_diff3 = self.pairwise_difference(grid, grid3)

        grid4 = grid
        grid4[:, 0] = grid[:, 0] + 1
        grid4[:, 1] = grid[:, 1] - 1
        pwd4 = sklearn.metrics.pairwise_distances(grid, grid4)
        X_diff4, Y_diff4 = self.pairwise_difference(grid, grid4)

        PWD = np.stack([pwd0, pwd1, pwd2, pwd3, pwd4], axis=2)
        X_DIFF = np.stack([X_diff0, X_diff1, X_diff2, X_diff3, X_diff4], axis=2)
        Y_DIFF = np.stack([Y_diff0, Y_diff1, Y_diff2, Y_diff3, Y_diff4], axis=2)
        pwd = np.min(PWD, axis=2)
        pwd_index = np.argmin(PWD, axis=2)
        edge_index = np.vstack(np.where(pwd <= self.radius))
        pwd_index = pwd_index[np.where(pwd <= self.radius)]
        PWD_index = (
            np.where(pwd <= self.radius)[0],
            np.where(pwd <= self.radius)[1],
            pwd_index,
        )
        distance = PWD[PWD_index]
        X_difference = X_DIFF[PWD_index]
        Y_difference = Y_DIFF[PWD_index]
        n_edges = edge_index.shape[1]
        return (
            torch.tensor(edge_index, dtype=torch.long),
            n_edges,
            distance,
            X_difference,
            Y_difference,
        )

    def get_data(self, theta):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        data = []
        for x in range(self.r):
            for y in range(self.r):
                grid_sub = self.grid[x :: self.r, y :: self.r, :].reshape(-1, 2)
                theta_sub = theta[x :: self.r, y :: self.r, :].reshape(-1, theta_d)

                perm = torch.randperm(self.n)
                m = self.m - grid_sub.shape[0]
                idx = perm[:m]
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                grid_split = torch.cat([grid_sub, grid_sample], dim=0)
                theta_split = torch.cat([theta_sub, theta_sample], dim=0)
                X = torch.cat([grid_split, theta_split], dim=1)

                edge_index, n_edges, distance, X_difference, Y_difference = (
                    self.torus_connectivity(grid_split)
                )

                edge_attr = np.zeros((n_edges, 3 + self.edge_features * 2))
                a = theta_split[:, : self.edge_features]
                edge_attr[:, 0] = X_difference.reshape(
                    n_edges,
                )
                edge_attr[:, 1] = Y_difference.reshape(
                    n_edges,
                )
                edge_attr[:, 2] = distance.reshape(
                    n_edges,
                )
                edge_attr[:, 3 : 3 + self.edge_features] = a[edge_index[0]]
                edge_attr[:, 3 + self.edge_features : 4 + self.edge_features * 2] = a[
                    edge_index[1]
                ]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)

                data.append(
                    Data(
                        x=X,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        split_idx=split_idx,
                    )
                )
        print("test", len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def sample(self, theta, Y, connectivity="ball"):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.resolution, self.resolution)

        x = torch.randint(0, self.r, (1,))
        y = torch.randint(0, self.r, (1,))

        grid_sub = self.grid[x :: self.r, y :: self.r, :].reshape(-1, 2)
        theta_sub = theta[x :: self.r, y :: self.r, :].reshape(-1, theta_d)
        Y_sub = Y[x :: self.r, y :: self.r].reshape(
            -1,
        )
        index_sub = self.index[x :: self.r, y :: self.r].reshape(
            -1,
        )
        n_sub = Y_sub.shape[0]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(
                self.n,
            )[idx]

            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=0).reshape(
                -1,
            )
            index_split = torch.cat([index_sub, idx], dim=0).reshape(
                -1,
            )
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(
                -1,
            )
            index_split = index_sub.reshape(
                -1,
            )
            X = torch.cat([grid_split, theta_split], dim=1)

        edge_index, n_edges, distance, X_difference, Y_difference = (
            self.torus_connectivity(grid_split)
        )

        edge_attr = np.zeros((n_edges, 3 + self.edge_features * 2))
        a = theta_split[:, : self.edge_features]
        edge_attr[:, 0] = X_difference.reshape(
            n_edges,
        )
        edge_attr[:, 1] = Y_difference.reshape(
            n_edges,
        )
        edge_attr[:, 2] = distance.reshape(
            n_edges,
        )
        edge_attr[:, 3 : 3 + self.edge_features] = a[edge_index[0]]
        edge_attr[:, 3 + self.edge_features : 4 + self.edge_features * 2] = a[
            edge_index[1]
        ]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        data = Data(
            x=X,
            y=Y_split,
            edge_index=edge_index,
            edge_attr=edge_attr,
            split_idx=split_idx,
            sample_idx=index_split,
        )
        print(
            "train",
            X.shape,
            Y_split.shape,
            edge_index.shape,
            edge_attr.shape,
            index_split.shape,
        )

        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.r**2 // batch_size2

        out = torch.zeros((self.resolution, self.resolution))
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i]
            for j in range(batch_size2):
                pred_ij = pred_i[j, :]
                x, y = split_idx_i[j]
                if self.resolution % 2 == 1:
                    if x == 0:
                        nx = self.s
                    else:
                        nx = self.s - 1
                    if y == 0:
                        ny = self.s
                    else:
                        ny = self.s - 1
                else:
                    nx = self.s
                    ny = self.s
                # pred_ij = pred_i[idx : idx + nx * ny]
                out[x :: self.r, y :: self.r] = pred_ij[: nx * ny].reshape(nx, ny)

        out = gaussian_filter(out, sigma=sigma, mode="constant", cval=0)
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(
            -1,
        )


def downsample(data, grid_size, l):
    data = data.reshape(-1, grid_size, grid_size)
    data = data[:, ::l, ::l]
    data = data.reshape(-1, (grid_size // l) ** 2)
    return data


def grid(n_x, n_y):
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if x != n_x - 1:
                edge_index.append((i, i + 1))
                edge_attr.append((1, 0, 0))
                edge_index.append((i + 1, i))
                edge_attr.append((-1, 0, 0))

            if y != n_y - 1:
                edge_index.append((i, i + n_x))
                edge_attr.append((0, 1, 0))
                edge_index.append((i + n_x, i))
                edge_attr.append((0, -1, 0))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge(n_x, n_y, a):
    a = a.reshape(n_x, n_y)
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if x != n_x - 1:
                d = 1 / n_x
                a1 = a[x, y]
                a2 = a[x + 1, y]
                edge_index.append((i, i + 1))
                edge_attr.append((d, a1, a2))
                edge_index.append((i + 1, i))
                edge_attr.append((d, a2, a1))

            if y != n_y - 1:
                d = 1 / n_y
                a1 = a[x, y]
                a2 = a[x, y + 1]
                edge_index.append((i, i + n_x))
                edge_attr.append((d, a1, a2))
                edge_index.append((i + n_x, i))
                edge_attr.append((d, a2, a1))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge_aug(n_x, n_y, a):
    a = a.reshape(n_x, n_y)
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if x != n_x - 1:
                d = 1 / n_x
                a1 = a[x, y]
                a2 = a[x + 1, y]
                edge_index.append((i, i + 1))
                edge_attr.append(
                    (
                        d,
                        a1,
                        a2,
                        1 / np.sqrt(np.abs(a1 * a2)),
                        np.exp(-((d) ** 2)),
                        np.exp(-((d / 0.1) ** 2)),
                        np.exp(-((d / 0.01) ** 2)),
                    )
                )
                edge_index.append((i + 1, i))
                edge_attr.append(
                    (
                        d,
                        a2,
                        a1,
                        1 / np.sqrt(np.abs(a1 * a2)),
                        np.exp(-((d) ** 2)),
                        np.exp(-((d / 0.1) ** 2)),
                        np.exp(-((d / 0.01) ** 2)),
                    )
                )

            if y != n_y - 1:
                d = 1 / n_y
                a1 = a[x, y]
                a2 = a[x, y + 1]
                edge_index.append((i, i + n_x))
                edge_attr.append(
                    (
                        d,
                        a1,
                        a2,
                        1 / np.sqrt(np.abs(a1 * a2)),
                        np.exp(-((d) ** 2)),
                        np.exp(-((d / 0.1) ** 2)),
                        np.exp(-((d / 0.01) ** 2)),
                    )
                )
                edge_index.append((i + n_x, i))
                edge_attr.append(
                    (
                        d,
                        a2,
                        a1,
                        1 / np.sqrt(np.abs(a1 * a2)),
                        np.exp(-((d) ** 2)),
                        np.exp(-((d / 0.1) ** 2)),
                        np.exp(-((d / 0.01) ** 2)),
                    )
                )

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge_aug_full(n_x, n_y, r, a):
    n = n_x * n_y

    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)

    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []

    for i1 in range(n):
        x1 = grid[i1]
        for i2 in range(n):
            x2 = grid[i2]

            d = np.linalg.norm(x1 - x2)

            if d <= r:
                a1 = a[i1]
                a2 = a[i2]
                edge_index.append((i1, i2))
                edge_attr.append(
                    (
                        d,
                        a1,
                        a2,
                        1 / np.sqrt(np.abs(a1 * a2)),
                        np.exp(-((d) ** 2)),
                        np.exp(-((d / 0.1) ** 2)),
                        np.exp(-((d / 0.01) ** 2)),
                    )
                )
                edge_index.append((i2, i1))
                edge_attr.append(
                    (
                        d,
                        a2,
                        a1,
                        1 / np.sqrt(np.abs(a1 * a2)),
                        np.exp(-((d) ** 2)),
                        np.exp(-((d / 0.1) ** 2)),
                        np.exp(-((d / 0.01) ** 2)),
                    )
                )

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def multi_grid(depth, n_x, n_y, grid, params):

    edge_index_global = []
    edge_attr_global = []
    X_global = []
    num_nodes = 0

    # build connected graph
    for l in range(depth):
        h_x_l = n_x // (2**l)
        h_y_l = n_y // (2**l)
        n_l = h_x_l * h_y_l

        a = downsample(params, n_x, (2**l))
        if grid == "grid":
            X, edge_index_inner, edge_attr_inner = grid(h_y_l, h_x_l)
        elif grid == "grid_edge":
            X, edge_index_inner, edge_attr_inner = grid_edge(h_y_l, h_x_l, a)
        elif grid == "grid_edge_aug":
            X, edge_index_inner, edge_attr_inner = grid_edge(h_y_l, h_x_l, a)

        # update index
        edge_index_inner = edge_index_inner + num_nodes
        edge_index_global.append(edge_index_inner)
        edge_attr_global.append(edge_attr_inner)

        # construct X
        # if (is_high):
        #     X = torch.cat([torch.zeros(n_l, l * 2), X, torch.zeros(n_l, (depth - 1 - l) * 2)], dim=1)
        # else:
        #     X_l = torch.tensor(l, dtype=torch.float).repeat(n_l, 1)
        #     X = torch.cat([X, X_l], dim=1)
        X_global.append(X)

        # construct edges
        index1 = torch.tensor(range(n_l), dtype=torch.long)
        index1 = index1 + num_nodes
        num_nodes += n_l

        # #construct inter-graph edge
        if l != depth - 1:
            index2 = np.array(range(n_l // 4)).reshape(
                h_x_l // 2, h_y_l // 2
            )  # torch.repeat is different from numpy
            index2 = index2.repeat(2, axis=0).repeat(2, axis=1)
            index2 = torch.tensor(index2).reshape(-1)
            index2 = index2 + num_nodes
            index2 = torch.tensor(index2, dtype=torch.long)

            edge_index_inter1 = torch.cat([index1, index2], dim=-1).reshape(2, -1)
            edge_index_inter2 = torch.cat([index2, index1], dim=-1).reshape(2, -1)
            edge_index_inter = torch.cat([edge_index_inter1, edge_index_inter2], dim=1)

            edge_attr_inter1 = torch.tensor((0, 0, 1), dtype=torch.float).repeat(n_l, 1)
            edge_attr_inter2 = torch.tensor((0, 0, -1), dtype=torch.float).repeat(
                n_l, 1
            )
            edge_attr_inter = torch.cat([edge_attr_inter1, edge_attr_inter2], dim=0)

            edge_index_global.append(edge_index_inter)
            edge_attr_global.append(edge_attr_inter)

    X = torch.cat(X_global, dim=0)
    edge_index = torch.cat(edge_index_global, dim=1)
    edge_attr = torch.cat(edge_attr_global, dim=0)
    mask_index = torch.tensor(range(n_x * n_y), dtype=torch.long)
    # print('create multi_grid with size:', X.shape,  edge_index.shape, edge_attr.shape, mask_index.shape)

    return (X, edge_index, edge_attr, mask_index, num_nodes)
