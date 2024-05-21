# Fourier-VAE
Using Fourier space properties to enhance inference speed and take advantage of paralellization.

# Convolutions in $\mathcal{F}$
$\mathcal{F}(f * g) = \mathcal{F}(f) \odot \mathcal{F}(g)$

# Deconvolutions in $\mathcal{F}$

$\mathcal{F}(f) = \frac{\mathcal{F}(f) \odot \mathcal{F}(g)}{\mathcal{F}(g) + \eps}$
