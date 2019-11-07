# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
# Internal imports
from .flow import Flow
from .layers import sum_dims, amortized_init


class PReLUFlow(Flow):
    """
    Normalizing flow version of a Parametric ReLU (PReLU).

    Contains one learnable parameter
        - alpha (slope of negative activation)
    """

    def __init__(self, dim, amortized='none'):
        super(PReLUFlow, self).__init__()
        self.amortized = amortized
        self.alpha = amortized_init(amortized, (1,))
        self.bijective = True

    def _call(self, z):
        """ Forward PReLU """
        return torch.where(z >= 0, z, torch.abs(self.alpha) * z)

    def _inverse(self, z):
        """ Inverse PReLU """
        return torch.where(z >= 0, z, torch.abs(1. / self.alpha) * z)

    def log_abs_det_jacobian(self, z):
        I = torch.ones_like(z)
        J = torch.where(z >= 0, I, self.alpha * I)
        log_abs_det = torch.log(torch.abs(J) + 1e-5)
        return sum_dims(log_abs_det)

    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        if self.amortized != 'none':
            self.alpha = params[:, :]

    def n_parameters(self):
        """ Return number of parameters in flow """
        return 1;


class SigmoidFlow(Flow):
    """
    Normalizing flow version of a Sigmoid
    """

    def __init__(self):
        super(SigmoidFlow, self).__init__()

    def _call(self, z):
        """ Forward sigmoid """
        return torch.sigmoid(z)

    def _inverse(self, z):
        """ Inverse of sigmoid is logit """
        return torch.log(z) - torch.log(1 - z)

    def log_abs_det_jacobian(self, z):
        return -sum_dims(F.softplus(z) + F.softplus(-z))


class LogitFlow(Flow):
    """
    Normalizing flow version of a Logit
    """

    def __init__(self):
        super(LogitFlow, self).__init__()

    def _call(self, z):
        """ Forward Logit """
        return torch.log(z) - torch.log(1 - z)

    def _inverse(self, z):
        """ Inverse of logit is sigmoid """
        return F.sigmoid(z)

    def log_abs_det_jacobian(self, z):
        return sum_dims(torch.log(z) + torch.log(-z + 1))
   
