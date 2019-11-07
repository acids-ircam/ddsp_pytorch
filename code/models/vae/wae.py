import torch
import torch.distributions as distrib
from models.vae.vae import VAE

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) 

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

class WAE(VAE):
    
    def __init__(self, encoder, decoder, input_dims, encoder_dims, latent_dims):
        super(WAE, self).__init__(encoder, decoder, input_dims, encoder_dims, latent_dims)
        
    def latent(self, x, z_params):
        n_batch = x.size(0)
        mu, log_var = z_params
        # Re-parametrize
        q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(log_var.shape[1]))
        #eps = q.sample((n_batch, )).detach().to(x.device)
        eps = torch.randn_like(mu).detach().to(x.device)
        z = (log_var.exp().sqrt() * eps) + mu
        # Sample from the z prior
        z_prior = q.sample((n_batch, )).to(x.device)
        # Compute MMD divergence
        mmd_dist = compute_mmd(z, z_prior)
        return z, mmd_dist
