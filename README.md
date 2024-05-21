
# Fourier Autoencoders
Using Fourier space properties to enhance inference speed and take advantage of paralellization.

## Convolutions in $\mathcal{F}$
$\mathcal{F}(f * g) = \mathcal{F}(f) \odot \mathcal{F}(g)$

## Deconvolutions in $\mathcal{F}$
$\mathcal{F}(f) = \frac{\mathcal{F}(f) \odot \mathcal{F}(g)}{\mathcal{F}(g) + \epsilon}$

## Algorithm backbone
### Encoder Nth step:
#### Identity CNN
We utilize a non-learnable convolutional neural network to project the cahnnel space into the desirable one.
#### Fourier Convolution
We use the convolution theorem to apply a weight mapping.
#### Further features
We can add complex activation functions in the fourier space, such as ReLU, SiLU, Sigmoid. We encourage the reader to keep an eye to the activation function utilization, given that this activation could bring instability in the fourier range due to the variance of the input image.
Also: batch, layer, grouped, instance normalization could be used in this case, It's a matter of trying them out.
## Autoencoder usage:
An autoencoder is usually a skip connection-like architecture between encoder and decoder. Here, the encoder-decoder skip connections would be the weights of the  
