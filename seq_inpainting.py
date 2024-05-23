# We here use the fundamentals of stable diffusion and make an embedding of 
# the prior time step to perform a normalized cross attention for each 
# upsampling in the decoder of the VAE
from torch import Tensor, nn
from utils import Encoder, UpsamplingFourierDeconvolution, FourierVAE
from framework.training.supervised import Module
from framework.nn.utils import residual_connection
from framework.nn.transformers.attention import CrossAttention, GroupedQueryAttention
from framework.nn.complex import Complex
from utils import Decoder as Decoder_
from typing import Sequence
from einops import rearrange
import torch
import random

class CrossAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cross_attention_block = Complex(CrossAttention(GroupedQueryAttention, )) # revise
        self.default = nn.Sequential(
            Complex(nn.GroupNorm(32, channels)),
            Complex(nn.SiLU)
        )
    def forward(self, input: Tensor, last_image: Tensor) -> Tensor:
        return residual_connection(input, lambda x: self.cross_attention_block(rearrange(x, 'b c h w -> b c (h w)'), rearrange(last_image, 'b c h w -> b c (h w)')))
    

class CrossUpsamplingBlock(UpsamplingFourierDeconvolution):
    def __init__(self, channels: int, height: int, width: int, pool: int, eps: float = 0.00001) -> None:
        super().__init__(channels, height, width, pool, eps)
        self.cross_attention = CrossAttentionBlock(channels)

    def forward(self, input: Tensor, E_t: Tensor, last_image: Tensor) -> Tensor:
        return self.cross_attention(super().forward(input, E_t), last_image)
    
class Decoder(Decoder_):
    def __init__(self, height: int, width: int, *channel_blocks) -> None:
        super().__init__(height, width, *channel_blocks)
        self.upsampling = nn.ModuleList([
            CrossUpsamplingBlock(channel, height/(2**i), width/(2**i), 2, 1e-6) for i, channel in reversed(enumerate(channel_blocks[1:]))
        ])

    def forward(self, x: Tensor, encoder_outputs: Sequence[Tensor], encoder_outputs_t_1: Sequence[Tensor]) -> Tensor:
        for layer, upsampling_block, encoder_output, last_image_output in zip(self.layers, self.upsampling, encoder_outputs[::-1], encoder_outputs_t_1[::-1]):
            x = layer(x)
            x = upsampling_block(x, encoder_output, last_image_output)
        return x


class CrossFourierVAE(FourierVAE):
    def __init__(self, *args, **hparams) -> None:
        super().__init__(**hparams)
        self.encoder = Encoder(256, 256, *args)
        self.decoder = Decoder(256, 256, *args)

    def forward(self, I_t: Tensor, I_t_1: Tensor) -> Tensor:
        masked = random.randint(0, 1)
        if masked:
            out_1, mu, logvar, hist = self.encoder.forward(I_t)

            with torch.no_grad():
                out_2, _, _, t_1_hist = self.encoder.forward(I_t_1)

            out = (out_1 + out_2) / 2

            out = self.decoder.forward(out, hist, t_1_hist)

            return out, mu, logvar
        
        out_1, mu, logvar, t_1_hist = self.encoder.forward(I_t_1)

        with torch.no_grad():
            out_2, _, _, hist = self.encoder.forward(I_t)

        out = (out_1 + out_2) / 2

        out = self.decoder.forward(out, hist, t_1_hist)

        return out, mu, logvar
    
    def loss_args(self, batch: Tensor) -> Tensor:
        I_gt, I_masked, I_t_1 = batch
        I_out, mu, logvar = self(I_masked, I_t_1)

        return I_out, I_gt, mu, logvar
                        
            
if __name__ == '__main__':
    # Test
    I_t = torch.randn(32, 3, 256, 256)
    I_t_1 = torch.randn(32, 3, 256, 256)

    model = CrossFourierVAE(3, 256, 256, 256, 256) # -> 64

    output = model(I_t, I_t_1)

    print(output)
    print('Done! All working as expected.')