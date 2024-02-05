import math

import torch
import torch.nn as nn
from e3nn import o3

from .utils import BroadcastGTOTensor


class GaussianOrbital(nn.Module):
    r"""
    Gaussian-type orbital

    .. math::
        \psi_{n\ell m}(\mathbf{r})=\sqrt{\frac{2(2a_n)^{\ell+3/2}}{\Gamma(\ell+3/2)}}
        \exp(-a_n r^2) r^\ell Y_{\ell}^m(\hat{\mathbf{r}})

    """

    def __init__(self, gauss_start, gauss_end, num_gauss, lmax=7):
        super(GaussianOrbital, self).__init__()
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.num_gauss = num_gauss
        self.lmax = lmax

        self.lc2lcm = BroadcastGTOTensor(lmax, num_gauss, src='lc', dst='lcm')
        self.m2lcm = BroadcastGTOTensor(lmax, num_gauss, src='m', dst='lcm')
        self.gauss: torch.Tensor
        self.lognorm: torch.Tensor

        self.register_buffer('gauss', torch.linspace(gauss_start, gauss_end, num_gauss))
        self.register_buffer('lognorm', self._generate_lognorm())

    def _generate_lognorm(self):
        power = (torch.arange(self.lmax + 1) + 1.5).unsqueeze(-1)  # (l, 1)
        numerator = power * torch.log(2 * self.gauss).unsqueeze(0) + math.log(2)  # (l, c)
        denominator = torch.special.gammaln(power)
        lognorm = (numerator - denominator) / 2
        return lognorm.view(-1)  # (l * c)

    def forward(self, vec):
        """
        Evaluate the basis functions
        :param vec: un-normalized vectors of (..., 3)
        :return: basis values of (..., (l+1)^2 * c)
        """
        # spherical
        device = vec.device
        r = vec.norm(dim=-1) + 1e-8
        spherical = o3.spherical_harmonics(
            list(range(self.lmax + 1)), vec / r[..., None],
            normalize=False, normalization='integral'
        )

        # radial
        r = r.unsqueeze(-1)
        lognorm = self.lognorm * torch.ones_like(r)  # (..., l * c)
        exponent = -self.gauss * (r * r)  # (..., c)
        poly = torch.arange(self.lmax + 1, dtype=torch.float, device=device) * torch.log(r)  # (..., l)
        log = exponent.unsqueeze(-2) + poly.unsqueeze(-1)  # (..., l, c)
        radial = torch.exp(log.view(*log.size()[:-2], -1) + lognorm)  # (..., l * c)
        return self.lc2lcm(radial) * self.m2lcm(spherical)  # (..., (l+1)^2 * c)

def fourier_basis(vec, num_fourier=4):
    # vec  (N, K, 3)
    device = vec.device
    N = torch.arange(num_fourier, dtype=torch.float, device=vec.device)
    N1,N2,N3 = torch.meshgrid(N, N, N)
    N = torch.stack([N1.flatten(),N2.flatten(),N3.flatten()],dim=1) # (f^3, 3)
    cos = torch.cos(vec.unsqueeze(-2) @ N.reshape(1,1,num_fourier**3,3) * math.pi).prod(-1) # (N, K, f^3)
    sin = torch.sin(vec.unsqueeze(-2) @ N.reshape(1,1,num_fourier**3,3) * math.pi).prod(-1) # (N, K, f^3)
    return torch.cat([cos,sin],dim=-1) # (N, K, 2*f^3)