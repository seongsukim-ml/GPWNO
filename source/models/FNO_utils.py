import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class SpectralConv3d_FFNO(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_FFNO, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes1
        self.modes_y = modes2
        self.modes_z = modes3

        self.fourier_weight = nn.ParameterList([])
        for n_modes in [self.modes_x, self.modes_y, self.modes_z]:
            weight = torch.FloatTensor(in_channels, out_channels, n_modes, 2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight.append(param)

    def forward(self, x):
        B, I, S1, S2, S3 = x.shape

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ftz = torch.fft.rfftn(x, dim=-1, norm="ortho")
        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        out_ft[:, :, :, :, : self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, : self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]),
        )
        xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm="ortho")

        xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm="ortho")
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-2, norm="ortho")
        out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1, S3)
        out_ft[:, :, :, : self.modes_y, :] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, : self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[1]),
        )

        xy = torch.fft.irfft(out_ft, n=S2, dim=-2, norm="ortho")

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-3, norm="ortho")
        out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        out_ft[:, :, : self.modes_x, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, : self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]),
        )
        xx = torch.fft.irfft(out_ft, n=S1, dim=-3, norm="ortho")

        # # Combining Dimensions # #
        x = xx + xy + xz

        return x


def plane_wave_sum(x):
    N = x.size(-1)
    n = torch.arange(N).float().to(x.device)
    k = n.view(-1, 1)
    M = torch.exp(-2j * np.pi * k * n)
    return torch.matmul(M, x)


# class SpectralConv2d(nn.Module):
#     def __init__(
#         self,
#         in_dim,
#         out_dim,
#         modes_x,
#         modes_y,
#         modes_z,
#         forecast_ff,
#         backcast_ff,
#         fourier_weight,
#         factor,
#         ff_weight_norm,
#         n_ff_layers,
#         layer_norm,
#         use_fork,
#         dropout,
#     ):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.modes_x = modes_x
#         self.modes_y = modes_y
#         self.modes_z = modes_z
#         self.use_fork = use_fork

#         self.fourier_weight = fourier_weight
#         # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
#         if not self.fourier_weight:
#             self.fourier_weight = nn.ParameterList([])
#             for n_modes in [modes_x, modes_y, modes_z]:
#                 weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
#                 param = nn.Parameter(weight)
#                 nn.init.xavier_normal_(param)
#                 self.fourier_weight.append(param)

#         if use_fork:
#             self.forecast_ff = forecast_ff
#             if not self.forecast_ff:
#                 self.forecast_ff = FeedForward(
#                     out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout
#                 )

#         self.backcast_ff = backcast_ff
#         if not self.backcast_ff:
#             self.backcast_ff = FeedForward(
#                 out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout
#             )

#     def forward(self, x):
#         # x.shape == [batch_size, grid_size, grid_size, in_dim]
#         x = self.forward_fourier(x)

#         b = self.backcast_ff(x)
#         f = self.forecast_ff(x) if self.use_fork else None
#         return b, f

#     def forward_fourier(self, x):
#         x = rearrange(x, "b s1 s2 s3 i -> b i s1 s2 s3")
#         # x.shape == [batch_size, in_dim, grid_size, grid_size]

#         B, I, S1, S2, S3 = x.shape

#         # # # Dimesion Z # # #
#         x_ftz = torch.fft.rfft(x, dim=-1, norm="ortho")
#         # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

#         out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
#         # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

#         out_ft[:, :, :, :, : self.modes_z] = torch.einsum(
#             "bixyz,ioz->boxyz",
#             x_ftz[:, :, :, :, : self.modes_z],
#             torch.view_as_complex(self.fourier_weight[2]),
#         )

#         xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm="ortho")
#         # x.shape == [batch_size, in_dim, grid_size, grid_size]

#         # # # Dimesion Y # # #
#         x_fty = torch.fft.rfft(x, dim=-2, norm="ortho")
#         # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

#         out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1, S3)
#         # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

#         out_ft[:, :, :, : self.modes_y, :] = torch.einsum(
#             "bixyz,ioy->boxyz",
#             x_fty[:, :, :, : self.modes_y, :],
#             torch.view_as_complex(self.fourier_weight[1]),
#         )

#         xy = torch.fft.irfft(out_ft, n=S2, dim=-2, norm="ortho")
#         # x.shape == [batch_size, in_dim, grid_size, grid_size]

#         # # # Dimesion X # # #
#         x_ftx = torch.fft.rfft(x, dim=-3, norm="ortho")
#         # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

#         out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
#         # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

#         out_ft[:, :, : self.modes_x, :, :] = torch.einsum(
#             "bixyz,iox->boxyz",
#             x_ftx[:, :, : self.modes_x, :, :],
#             torch.view_as_complex(self.fourier_weight[0]),
#         )

#         xx = torch.fft.irfft(out_ft, n=S1, dim=-3, norm="ortho")
#         # x.shape == [batch_size, in_dim, grid_size, grid_size]

#         # # Combining Dimensions # #
#         x = xx + xy + xz

#         x = rearrange(x, "b i s1 s2 s3 -> b s1 s2 s3 i")
#         # x.shape == [batch_size, grid_size, grid_size, out_dim]

#         return x
