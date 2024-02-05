import io

import PIL
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, CenteredNorm
import matplotlib.colors as colors
from torchvision.transforms import ToTensor

plt.switch_backend('agg')
cmap = ListedColormap(['grey', 'white', 'red', 'blue', 'green', 'white'])
norm = CenteredNorm

def draw_stack(density, atom_type=None, atom_coord=None, dim=-1):
    """
    Draw a 2D density map along specific axis.
    :param density: density data, tensor of shape (batch_size, nx, ny, nz)
    :param atom_type: atom types, tensor of shape (batch_size, n_atom)
    :param atom_coord: atom coordinates, tensor of shape (batch_size, n_atom, 3)
    :param dim: axis along which to sum
    :return: an image tensor
    """
    plt.figure(figsize=(3, 3))
    plt.imshow(density.sum(dim).detach().cpu().numpy(), cmap='bwr', norm=colors.CenteredNorm())
    plt.colorbar()
    if atom_type is not None:
        idx = [i for i in range(3) if i != dim % 3]
        coord = atom_coord.detach().cpu().numpy()
        color = cmap(atom_type.detach().cpu().numpy())
        plt.scatter(coord[:, idx[1]], coord[:, idx[0]], c=color, alpha=0.8)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close()
    return image

def draw_stack_probe(density, atom_type=None, atom_coord=None, dim=-1, probe=None):
    """
    Draw a 2D density map along specific axis.
    :param density: density data, tensor of shape (batch_size, nx, ny, nz)
    :param atom_type: atom types, tensor of shape (batch_size, n_atom)
    :param atom_coord: atom coordinates, tensor of shape (batch_size, n_atom, 3)
    :param dim: axis along which to sum
    :return: an image tensor
    """
    plt.figure(figsize=(3, 3))
    plt.imshow(density.sum(dim).detach().cpu().numpy(), cmap='bwr',norm=colors.CenteredNorm())
    plt.colorbar()
    if atom_type is not None:
        idx = [i for i in range(3) if i != dim % 3]
        coord = atom_coord.detach().cpu().numpy()
        color = cmap(atom_type.detach().cpu().numpy())
        plt.scatter(coord[:, idx[1]], coord[:, idx[0]], c=color, alpha=0.8)
    
    # if probe is not None:
    #     idx = [i for i in range(3) if i != dim % 3]
    #     coord = probe.detach().cpu().numpy()
    #     plt.scatter(coord[:, idx[1]], coord[:, idx[0]], c="m", s=2,alpha=0.008)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close()
    return image
