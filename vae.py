from framework.nn.complex import Complex
from framework.nn.fourier import FourierConv2d, FourierDeconv2d
from framework.nn.functional import residual_connection
from framework.training.supervised import Module
from torch import Tensor, nn
import torch
import torch.nn.functional as F
import math

VALID_ACTIVATIONS = {
    'relu': Complex(nn.ReLU),
    'sigmoid': Complex(nn.Sigmoid),
    'gelu': Complex(nn.GELU),
    'silu': Complex(nn.SiLU),
    'None': nn.Identity
}

class IdentityConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: int = 2) -> None:
        super().__init__()
        self.pool = pool
        self.weight = torch.ones(out_channels, in_channels, 1, 1, requires_grad=False)
    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input.real, self.weight, stride = self.pool) + 1j*F.conv2d(input.imag, self.weight, stride = self.pool)

class DownsamplingFourierConvolution(nn.Sequential):
    def __init__(
            self,
            channels: int,
            height: int,
            width: int,
            pool: int
    ) -> None:
        super().__init__(
            Complex(nn.GroupNorm, 32, channels),
            Complex(nn.SiLU),
            FourierConv2d(channels, channels, height, width),
            Complex(nn.MaxPool2d, pool, pool)
        )
    
    def forward(self, input):
        return super().forward(input)
    
class ResidualBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, height: int, width: int) -> None:
        super().__init__()
        if in_channels == out_channels:
            self.residue = nn.Identity()
        else:
            self.residue = FourierConv2d(in_channels, out_channels, height, width, bias = True)

        self.sublayer = nn.Sequential(
            Complex(nn.GroupNorm, 32, in_channels),
            Complex(nn.SiLU),
            FourierConv2d(in_channels, out_channels, height, width, bias = True),
            Complex(nn.GroupNorm, 32, out_channels),
            Complex(nn.SiLU),
            FourierConv2d(out_channels, out_channels, height, width, bias = True)
            )
    
    def forward(self, input: Tensor) -> Tensor:
        return self.residue(input) + self.sublayer(input)

class UpsamplingFourierDeconvolution(nn.Sequential):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        pool: int,
        eps: float = 0.00001
    ) -> None:
        super().__init__()
        self.encoder_norm = nn.Sequential(
            Complex(nn.GroupNorm, 32, channels),
            Complex(nn.SiLU),
        )
        self.decoder_norm = nn.Sequential(
            Complex(nn.GroupNorm, 32, channels),
            Complex(nn.SiLU),
        )

        self.pool = Complex(nn.UpsamplingBilinear2d, scale_factor = pool)

        self.weight = torch.fft.fftn(
            nn.init.kaiming_normal_(
                nn.Parameter(
                    torch.empty((channels, channels, height, width)),
                    True
                ),
                math.sqrt(5),
            )
        )

        self.bias = nn.init.kaiming_normal_(
                nn.Parameter(
                    torch.empty((1, channels, 1, 1)),
                    True
                ),
                math.sqrt(5),
            )
        
        self.eps = eps
    def forward(self, input: Tensor, E_t: Tensor) -> Tensor:
        return ((self.encoder_norm(E_t) + self.decoder_norm(input))/(self.weight + self.eps)) + self.bias.view(1, self.channels, 1, 1)

class FourierVAE(Module):
    def __init__(self, **hparams) -> None:
        # Setting them up
        for key in hparams:
            setattr(self, key, hparams[key])
        self.save_hyperparameters()

    def loss_args(self, batch: Tensor) -> Tensor:
        I_gt, mask_in = batch
        I_out, mu, logvar = self(I_gt, mask_in)
        return I_out, I_gt, mu, logvar

    def validation_step(self, batch: Tensor, idx) -> Tensor:
        I_gt, mask_in = batch
        I_out, mu, logvar = self(I_gt, mask_in)
        args = self.criterion(I_out, I_gt, mu, logvar)
        self.log_dict(
            {f"Validation/{k}": v for k, v in zip(self.criterion.labels, args)}
        )
        self.log('hp_metric', args[-1])

from lightning.pytorch import LightningModule