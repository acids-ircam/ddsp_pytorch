# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from torch import distributions as distrib
from torch.distributions import MultivariateNormal, transforms as transform

class Flow(transform.Transform, nn.Module):

    """
    Main class for a single flow.
    """

    def __init__(self, amortized='none'):
        """ Initialize as both transform and module """
        transform.Transform.__init__(self)
        nn.Module.__init__(self)
        # Handle amortization
        self.amortized = amortized

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.001, 0.001)

    def __hash__(self):
        """ Dirty hack to ensure nn.Module compatibility """
        return nn.Module.__hash__(self)

    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        pass

    def n_parameters(self):
        """ Return number of parameters in flow """
        return 0


class FlowList(nn.ModuleList, Flow):#, transform.ComposeTransform):

    """Concatenation of several flows."""

    def __init__(self, flows=[]):
        Flow.__init__(self, flows[0].amortized if flows else 'none')
        nn.ModuleList.__init__(self, flows)
        #transform.ComposeTransform.__init__(self, flows)

    def _call(self, z):
        for flow in self:
            z = flow(z)
        return z

    def append(self, flow):
        nn.ModuleList.append(self, flow)
        #self.parts = list(self.parts) + [flow]

    def log_abs_det_jacobian(self, z):
        if not self:
            return torch.zeros_like(z)
        result = 0
        for flow in self:
            result = result + flow.log_abs_det_jacobian(z)
        return result

    def n_parameters(self):
        return sum(flow.n_parameters() for flow in self)

    def set_parameters(self, params, batch_dim=1):
        i = 0
        for flow in self:
            j = i + flow.n_parameters()
            flow.set_parameters(params[:, i:j], batch_dim)
            i = j


class NormalizingFlow(nn.Module):

    """
    Main class for a normalizing flow, defined as a sequence of flows.

    The flows is defined as a set of blocks, each of which can contain
    several flows.
    This sequence is contained in the `bijectors` list.
    We also define the following properties for flows
        - dim
        - base_density (Distribution)
        - transforms (Transform)
        - bijectors (ModuleList)
        - final_density (TransformedDistribution)
        - amortized (str)
            Defines the type of amortization, which takes values in
            'none'  : Each flow manages its own parameters
            'self'  : The flow itself will provide parameters
            'input' : The parameters are external to the flow
    """

    def __init__(self, dim, blocks, flow_length, final_block=None, density=None, amortized='none'):
        """ Initialize normalizing flow """
        super().__init__()
        biject = []
        self.n_params = []
        # Start density (z0)
        if density is None:
            density = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        self.base_density = density
        for f in range(flow_length-1):
            for b_flow in blocks:
                cur_block = b_flow(dim, amortized=amortized)
                self.n_params.append(cur_block.n_parameters())
                biject.append(cur_block)
        # Add only first block last
        cur_block = blocks[0](dim, amortized=amortized)
        self.n_params.append(cur_block.n_parameters())
        biject.append(cur_block)
        if (final_block is not None):            
            cur_block = final_block
            self.n_params.append(cur_block.n_parameters())
            biject.append(cur_block)
        # Full set of transforms
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        # Final density (zK) defined as transformed distribution
        self.final_density = distrib.TransformedDistribution(density, self.transforms)
        self.amortized = amortized
        # Handle different amortizations
        if amortized in ('self', 'input'):
            self.amortized_seed = torch.ones(1, dim).detach()
            self.amortized_params = self.parameters_network(dim, self.n_parameters())
        self.log_det = []
        self.dim = dim

    def parameters_network(self, nin, nout, nlayers=2, nhidden=64, activation=nn.ReLU):
        net = nn.ModuleList()
        for l in range(nlayers):
            cur_n = nn.Linear((l==0) and nin or nhidden, (l==nlayers-1) and nout or nhidden)
            nn.init.xavier_uniform_(cur_n.weight.data, gain=.1)
            cur_n.bias.data.uniform_(0,0)
            net.append(cur_n)
            if (l < nlayers - 1):
                net.append(activation())
        return nn.Sequential(*net)

    def amortization(self, z):
        if self.amortized in ['none', 'ext']:
            return
        elif self.amortized == 'self':
            if self.amortized_seed.device != z.device:
                self.amortized_seed = self.amortized_seed.to(z.device)
            params = self.amortized_params(self.amortized_seed)
        elif self.amortized == 'input':
            params = self.amortized_params(z)
        self.set_parameters(params, z.shape[0])

    def forward(self, z):
        """ Forward a set of samples and apply series of flows """
        self.log_det = []
        self.amortization(z)
        for bijector in self.bijectors:
            output = bijector(z)
            self.log_det.append(bijector.log_abs_det_jacobian(z))
            z = output
        return z, self.log_det

    def inverse(self, z):
        """ Apply inverse series of flows to a set of samples """
        self.log_det = []
        self.amortization(z)
        for b in range(len(self.bijectors)-1, 0, -1):
            z = self.bijectors[b].inv(z)
        return z, 0

    def n_parameters(self):
        """ Total number of parameters for all flows """
        return sum(self.n_params)

    def set_parameters(self, params, batch_dim=64):
        """ Set the flows parameters """
        param_list = params.split(self.n_params, dim=1)
        for params, bijector in zip(param_list, self.bijectors):
            bijector.set_parameters(params, batch_dim)


class NormalizingFlowContext(NormalizingFlow):

    """
    Main class for amortized normalizing flows.

    Very similar to a NormalizingFlow. However, in this case the parameters are
    produced through a context variable that is passed at the forward call.
    Hence external forward calls should give a tuple (input, context).

    The principle of amortization is that parameters are computed by external
    neural networks and are different for each input. In this case, we will use
    the same context variable to produce the parameters and context
    """

    def forward(self, inputs):
        """ Forward a set of samples and apply series of flows """
        self.log_det = []
        # Retrieve context
        z, ctx = inputs
        if self.amortized in ['self', 'input']:
            params = self.amortized_params(ctx)
            self.set_parameters(params, z.shape[0])
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian((z, ctx)))
            z = self.bijectors[b]((z, ctx))
        return z, self.log_det


class GenerativeFlow(NormalizingFlow):

    """
    Main class for a generative flow, defined as a sequence of flows.

    Oppositely to a typical flow (defined as a set of blocks), here the generative
    flow is defined as a sequence of a repeated set that contains several flows.
    This sequence is contained in the `bijectors` list.
    """

    def __init__(self, dim, blocks, generative_layers, args, target_density=distrib.MultivariateNormal, learn_top=False, y_condition=False):
        """ Initialize normalizing flow """
        super(GenerativeFlow, self).__init__(dim, blocks, generative_layers, target_density, 'none')
        biject = []
        self.n_params = []
        self.output_shapes = []
        self.target_density = target_density
        # Get input size
        C, H, W = args.input_size
        # Create the L layers
        for l in range(generative_layers):
            C, H, W = C * 4, H // 2, W // 2
            self.output_shapes.append([-1, C, H, W])
            for b_flow in blocks:
                cur_block = b_flow(C, amortized='none')
                biject.append(cur_block)
                self.n_params.append(cur_block.n_parameters())
            C = C // 2
        C, H, W = C * 4, H // 2, W // 2
        self.output_shapes.append([-1, C, H, W])
        # Add a last layer (avoiding last block)
        for b_flow in blocks[:-1]:
            cur_block = b_flow(C, amortized='none')
            biject.append(cur_block)
            self.n_params.append(cur_block.n_parameters())
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.final_density = distrib.TransformedDistribution(target_density, self.transforms)
        self.dim = dim
        # self.y_classes = hparams.Glow.y_classes
        self.learn_top = learn_top
        self.y_condition = y_condition
        # for prior
        if self.learn_top:
            self.top_layer = nn.Conv2d(C * 2, C * 2)
        if self.y_condition:
            self.project_ycond = LinearZeros(y_classes, 2 * C)
            self.project_class = LinearZeros(C, y_classes)
        # Register learnable prior
        self.prior_h = nn.Parameter(torch.zeros([args.batch_size, C * 2, H, W]))

    def prior(self, y_onehot=None):
        B, C = self.prior_h.size(0), self.prior_h.size(1)
        h = self.prior_h.detach().clone()
        assert torch.sum(h) == 0.0
        if self.learn_top:
            h = self.top_layer(h)
        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot).view(B, C, 1, 1)
            h += yp
        return h.chunk(2, dim=1)

    def sample(self, x, n_samples=64):
        self.eval()
        z = self.target_density.sample((n_samples, ))
        return self.inverse(z)

    def forward(self, z):
        """ Forward a set of samples and apply series of flows """
        self.log_det = []
        # Size of input
        dim_in = z.shape[2] * z.shape[3]
        # Add noise to input
        z = z + torch.uniform(mean=torch.zeros_like(z), std=torch.ones_like(z) * (1. / 256.))
        # Include noise in the log_dets
        self.log_det.append(torch.zeros_like(z[:, 0, 0, 0]) + (-np.log(256.) * dim_in))
        #self.log_det.append((-np.log(256.) * dim_in))
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z, self.log_det

    def inverse(self, z):
        """ Apply inverse series of flows to a set of samples """
        self.log_det = []
        # Applies reverse series of flows
        for b in range(len(self.bijectors)-1, 0, -1):
            z = self.bijectors[b].inv(z)
        return z

    def n_parameters(self):
        """ Total number of parameters for all flows """
        return sum(self.n_params)

