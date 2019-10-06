# -*- coding: utf-8 -*-
import torch
from torch import nn
# Internal imports
from .flow import Flow

class SylvesterFlow(Flow):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, dim, num_ortho_vecs=16, steps=50, amortized='none'):
        """
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param q_ortho: shape (batch_size, z_size, num_ortho_vecs)
        :param b: shape: (batch_size, 1, self.z_size)
        """
        super(SylvesterFlow, self).__init__()
        self.num_ortho_vecs = num_ortho_vecs
        self.h = nn.Tanh()
        self.steps = steps
        self.amortized = amortized
        self.diag_idx = torch.arange(0, num_ortho_vecs).long()
        self._eye = torch.eye(self.num_ortho_vecs, self.num_ortho_vecs).unsqueeze(0).detach()
        self.dim = dim
        self.zk = []
        self.r1 = []
        self.r2 = []
        self.q_ortho = []
        self.b = []
        # Register buffers
        self.register_buffer('eye', self._eye)
        self.register_buffer('idx', self.diag_idx)
        

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2
        
    def _call(self, z):
        """
        All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :return: z, log_det_j
        """
        # Amortized flow parameters
        zk = z.unsqueeze(1)
        # Keep r1 and r2
        r1_hat = self.r1
        r2_hat = self.r2
        # Compute QR1 and QR2
        qr2 = torch.bmm(self.q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(self.q_ortho, r1_hat)
        # R2Q^T z + b
        r2qzb = torch.bmm(zk, qr2) + self.b
        z = torch.bmm(self.h(r2qzb), qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)
        return z

    def _inverse(self, z):
        raise Exception('Not implemented')

    def log_abs_det_jacobian(self, z):
        # Amortized flow parameters
        zk = z.unsqueeze(1)
        # Save diagonals for log_det_j
        diag_r1 = self.r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = self.r2[:, self.diag_idx, self.diag_idx]
        # Keep r1 and r
        r2_hat = self.r2
        # Compute QR1 and QR2
        qr2 = torch.bmm(self.q_ortho, r2_hat.transpose(2, 1))
        # R2Q^T z + b
        r2qzb = torch.bmm(zk, qr2) + self.b
        # Compute log|det J|
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_det_j = diag_j.abs().log()
        return torch.sum(log_det_j).repeat(z.shape[0], 1)
    
    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        if (self.amortized != 'none'):
            self.alpha = params[:, :]
        self.zk = params[:, :self.dim].repeat(batch_dim, 1)
        self.r1 = params[:, self.dim:(self.dim + self.num_ortho_vecs**2)].repeat(batch_dim, 1).view(-1, self.num_ortho_vecs, self.num_ortho_vecs)
        self.r2 = params[:, (self.dim + self.num_ortho_vecs**2):(self.dim + (2 * self.num_ortho_vecs**2))].repeat(batch_dim, 1).view(-1, self.num_ortho_vecs, self.num_ortho_vecs)
        self.q_ortho = params[:, (self.dim + (2 * self.num_ortho_vecs**2)):(self.dim + (2 * self.num_ortho_vecs**2) + (self.dim * self.num_ortho_vecs))].repeat(batch_dim, 1).view(-1, self.dim, self.num_ortho_vecs)
        self.b = params[:, (self.dim + (2 * self.num_ortho_vecs**2) + (self.dim * self.num_ortho_vecs)):].repeat(batch_dim, 1).view(-1, 1, self.num_ortho_vecs)
        # Handle orthogonalization of Q
        self.q_ortho = self.construct_orthogonal(self.q_ortho)
    
    def n_parameters(self):
        """ Return number of parameters in flow """
        return (self.dim) + (2 * self.num_ortho_vecs**2) + ((self.dim + 1) * self.num_ortho_vecs)
    
    def construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size * num_flows, z_size * num_ortho_vecs)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, z_size, num_ortho_vecs)
        """
        # Reshape to shape (num_flows * batch_size, z_size * num_ortho_vecs)
        q = q.view(-1, self.dim * self.num_ortho_vecs)
        # Constroct norms
        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        amat = torch.div(q, norm)
        dim0 = amat.size(0)
        amat = amat.resize(dim0, self.dim, self.num_ortho_vecs)
        # Stopping criterion
        max_norm = 0.
        # Iterative orthogonalization
        for s in range(self.steps):
            tmp = torch.bmm(amat.transpose(2, 1), amat)
            tmp = self._eye - tmp
            tmp = self._eye + 0.5 * tmp
            amat = torch.bmm(amat, tmp)
            # Testing for convergence
            test = torch.bmm(amat.transpose(2, 1), amat) - self._eye
            norms2 = torch.sum(torch.norm(test, p=2, dim=2) ** 2, dim=1)
            norms = torch.sqrt(norms2)
            max_norm = torch.max(norms).item()
            if max_norm <= 1e-5:
                break
        # Reshaping: first dimension is batch_size
        amat = amat.view(-1, self.dim, self.num_ortho_vecs)
        return amat

class TriangularSylvesterFlow(SylvesterFlow):
    """
    Triangular Sylvester normalizing flow.
    """

    def __init__(self, dim, num_ortho_vecs=16, steps=50, amortized='none'):
        """
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, z_size, z_size)
        :param r2: shape: (batch_size, z_size, z_size)
        :param b: shape: (batch_size, 1, z_size)
        """
        super(TriangularSylvesterFlow, self).__init__(dim, num_ortho_vecs, steps, amortized)
        self.num_ortho_vecs = num_ortho_vecs
        self.diag_idx = torch.arange(0, self.dim).long()
        self.mask = torch.triu(torch.ones(self.dim, self.dim), diagonal=1).unsqueeze(0)
        # Register buffers
        self.register_buffer('idx_d', self.diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2
        
    def _call(self, z):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :return: z, log_det_j
        """
        # Amortized flow parameters
        zk = z.unsqueeze(1)
        # R2Q^T z + b
        r2qzb = torch.bmm(zk, self.r2.transpose(2, 1)) + self.b
        z = torch.bmm(self.h(r2qzb), self.r1.transpose(2, 1)) + zk
        z = z.squeeze(1)
        return z

    def _inverse(self, z):
        raise Exception('Not implemented')

    def log_abs_det_jacobian(self, z):
        # Amortized flow parameters
        zk = z.unsqueeze(1)
        # Save diagonals for log_det_j
        diag_r1 = self.r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = self.r2[:, self.diag_idx, self.diag_idx]
        # R2Q^T z + b
        r2qzb = torch.bmm(zk, self.r2.transpose(2, 1)) + self.b
        # Compute log|det J|
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_det_j = diag_j.abs().log()
        return torch.sum(log_det_j).repeat(z.shape[0], 1)
    
    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        if (self.amortized != 'none'):
            self.alpha = params[:, :]
        self.mask = self.mask.to(params.device)
        full_d = params[:, :(self.dim**2)].view(-1, self.dim, self.dim)
        diag1 = params[:, (self.dim**2):((self.dim**2) + self.dim)].view(-1, self.dim)
        diag2 = params[:, ((self.dim**2) + self.dim):((self.dim**2) + (2 * self.dim))].view(-1, self.dim)
        self.b = params[:, ((self.dim**2) + (2 * self.dim)):].view(-1, 1, self.dim)#.repeat(batch_dim, 1).view(-1, 1, self.dim)
        self.r1 = full_d * self.mask
        self.r2 = full_d.transpose(2, 1) * self.mask
        self.r1[:, self.diag_idx, self.diag_idx] = diag1
        self.r2[:, self.diag_idx, self.diag_idx] = diag2
        if (self.amortized == 'self'):
            self.r1 = self.r1.repeat(batch_dim, 1, 1)
            self.r2 = self.r2.repeat(batch_dim, 1, 1)
            self.b = self.b.repeat(batch_dim, 1, 1)
    
    def n_parameters(self):
        """ Return number of parameters in flow """
        return (self.dim * 3) + (self.dim**2)

