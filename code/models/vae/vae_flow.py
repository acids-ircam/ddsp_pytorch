import torch
import torch.nn as nn
from models.vae.vae import VAE

class VAEFlow(VAE):
    
    def __init__(self, encoder, decoder, flow, input_dims, encoder_dims, latent_dims):
        super(VAEFlow, self).__init__(encoder, decoder, input_dims, encoder_dims, latent_dims)
        self.flow_enc = nn.Linear(encoder_dims, flow.n_parameters())
        self.flow = flow
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            m.weight.data.uniform_(-0.01, 0.01)
            m.bias.data.fill_(0.0)
            
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        flow_params = self.flow_enc(x)
        return mu, log_var, flow_params

    def latent(self, x, z_params):
        n_batch = x.size(0)
        # Split the encoded values to retrieve flow parameters
        mu, log_var, flow_params = z_params
        # Re-parametrize a Normal distribution
        eps = torch.randn_like(mu).detach().to(x.device)
        # Obtain our first set of latent points
        z_0 = (log_var.exp().sqrt() * eps) + mu
        # Update flows parameters
        self.flow.set_parameters(flow_params)
        # Complexify posterior with flows
        z_k, list_ladj = self.flow(z_0)
        # ln p(z_k) 
        log_p_zk = torch.sum(-0.5 * z_k * z_k, dim=1)
        # ln q(z_0)  (not averaged)
        log_q_z0 = torch.sum(-0.5 * (log_var + (z_0 - mu) * (z_0 - mu) * log_var.exp().reciprocal()), dim=1)
        # ln q(z_0) - ln p(z_k)
        logs = (log_q_z0 - log_p_zk).sum()
        # Add log determinants
        ladj = torch.cat(list_ladj, dim=1)
        # ln q(z_0) - ln p(z_k) - sum[log det]
        logs -= torch.sum(ladj)
        #print(logs)
        return z_k, (logs / float(n_batch))
