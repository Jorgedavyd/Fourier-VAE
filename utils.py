from framework.nn.complex import Complex
from framework.nn.fourier import FourierConv2d
from framework.nn.functional import residual_connection
from framework.training.supervised import Module
from framework.nn.transformers.attention import SelfAttention, GroupedQueryAttention
from einops import rearrange
from torch import Tensor, nn
import torch
import torch.nn.functional as F
import math
from typing import Sequence, List
from loss import Loss

VALID_ACTIVATIONS = {
    'relu': Complex(nn.ReLU),
    'sigmoid': Complex(nn.Sigmoid),
    'gelu': Complex(nn.GELU),
    'silu': Complex(nn.SiLU),
    'None': nn.Identity
}

class AttentionBlock(nn.Sequential):
    def __init__(
            self,
            channels: int
    ) -> None:
        super().__init__(
            Complex(nn.GroupNorm(32, channels)),
            Complex(nn.SiLU),
            Complex(SelfAttention(GroupedQueryAttention))
        )
        self.default_step = nn.Sequential(
            Complex(nn.GroupNorm(32, channels)),
            Complex(nn.SiLU)
        )
        self.attention = Complex(SelfAttention(GroupedQueryAttention()))

    def forward(self, x: Tensor) -> Tensor:
        return residual_connection(
            x,
            lambda x: self.attention(rearrange(self.default_step(x), 'b c h w -> b c (h w)')).reshape(x.shape)
            )
    
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
            Complex(nn.MaxPool2d, pool, pool),
            FourierConv2d(channels, channels, height, width),
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
        input = self.pool(input)
        out = self.encoder_norm(E_t) + self.decoder_norm(input) # Skip connection
        out = out / (self.weight + self.eps) # Deconvolution step
        out += self.bias
        return out
    

class Encoder(nn.Module):
    def __init__(self, height: int, width: int, *channel_blocks) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(in_channels, out_channels, height/(2**i), width/(2**i)),
                AttentionBlock(out_channels)
            ) for i, in_channels, out_channels in enumerate(zip(channel_blocks[:-1], channel_blocks[1:]))
        ])

        self.downsampling = nn.ModuleList([
            DownsamplingFourierConvolution(channel, height/(2**(i+1)), width/(2**(i+1)), 2, 1e-6) for i, channel in enumerate(channel_blocks[1:])
        ])

        self.num_layers = len(channel_blocks)
        self.channels = channel_blocks

        self.last_height, self.last_width = height/(2**len(channel_blocks)), width/(2**len(channel_blocks))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_blocks[-1] * (height*width)/(4**(2*len(channel_blocks))), (height*width)/(4**(2*len(channel_blocks))))
        )

    def reparametrization(self, x: Tensor) -> Tensor:
        # Getting the batch size
        b, _, _ = x.shape
        # Forward the projection to the distribution parameter space
        out = self.fc(x)
        # Getting the mu and log variance
        mu, log_variance = torch.chunk(out, 2, -1)
        # Constraining
        log_variance = torch.clamp(log_variance, -30, 20)
        # Standard deviation
        std = log_variance.exp().sqrt()
        # Transforming the normal distribution into the given parameter space
        out = mu + torch.randn(b, self.channels[-1], self.last_height, self.last_width) * std

        return out*0.18215, mu, log_variance

    def forward(self, x: Tensor) -> Tensor:
        hist: List[Tensor] = []
        for layer, upsampling_block in zip(self.layers, self.upsampling):
            x = layer(x)
            hist.append(x) # 1024, 512, 256, ...
            x = upsampling_block(x)
        out, mu, logvar = self.reparametrization(x)
        return out, mu, logvar, hist
    
class Decoder(nn.Module):
    def __init__(self, height: int, width: int, *channel_blocks) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(in_channels, out_channels, height/(2**i), width/(2**i)),
                AttentionBlock(out_channels)
            ) for i, in_channels, out_channels in reversed(enumerate(zip(channel_blocks[:-1], channel_blocks[1:])))
        ])

        self.upsampling = nn.ModuleList([
            UpsamplingFourierDeconvolution(channel, height/(2**i), width/(2**i), 2, 1e-6) for i, channel in reversed(enumerate(channel_blocks[1:]))
        ])

        self.num_layers = len(channel_blocks)

    def forward(self, x: Tensor, encoder_outputs: Sequence[Tensor]) -> Tensor:
        for layer, upsampling_block, encoder_output in zip(self.layers, self.upsampling, encoder_outputs[::-1]):
            x = layer(x)
            x = upsampling_block(x, encoder_output)
        return x

class FourierVAE(Module):
    def __init__(self, *args, **hparams) -> None:
        super().__init__(**hparams)
        self.encoder = Encoder(256, 256, *args)
        self.decoder = Decoder(256, 256, *args)
        self.criterion = Loss(self.alpha, self.beta)
    def forward(self, x: Tensor) -> Tensor:
        x, mu, logvar, hist = self.encoder(x)
        x = self.decoder(x, hist)
        return x, mu, logvar

    def loss_args(self, batch: Tensor) -> Tensor:
        I_gt, I_masked = batch
        I_out, mu, logvar = self(I_masked)
        return I_out, I_gt, mu, logvar


