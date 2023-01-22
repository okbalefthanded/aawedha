# Okba Bekhelifi <okba.bekhelifi@univ-usto.dz> https://github.com/okbalefthanded

# Implementation of Trainable Weight Averaging (TWA) [1] following the official
# PyTorch Stochatsic Weight Averaging (SWA) [2] class.

# References:
# [1] Li, T. et al. (2022) ‘Trainable Weight Averaging for Fast 
# Convergence and Better Generalization’, pp. 1–14. Available at: http://arxiv.org/abs/2205.13104.
#
# [2] Izmailov, P. et al. (2018) ‘Averaging weights leads to wider 
# optima and better generalization’, 34th Conference on Uncertainty 
# in Artificial Intelligence 2018, UAI 2018, 2, pp. 876–885.

# This implementation is based on the official TWA code:
# https://github.com/nblt/TWA

import torch


class TWA:
    def __init__(self, device=None):
        super().__init__() 
        self.P = None 
        self.W = []
        self.n_averaged = 0
        self.device = device
        self.fitted = False

    def update_parameters(self, model):
        gk = self.get_model_grad_vec(model)        
        self.P_SGD(model, gk)

    def P_SGD(self, model, grad):
        gk = torch.mm(self.P, grad.reshape(-1, 1))
        grad_proj = torch.mm(self.P.transpose(0, 1), gk)    
        self.update_grad(model, grad_proj.reshape(-1))

    def collect_solutions(self, model):
        self.W.append(self.get_model_param_vec(model))
        self.n_averaged += 1
        # W = np.array(W)

    def fit_subspace(self):
        # orthogonalize W to e_i and create P = [e_1, e_2...e_n]
        # self.P = torch.tensor(self.W, device=self.device)
        self.P = torch.stack(self.W)
        n_dim = self.P.shape[0]
        for i in range(n_dim):
            if i > 0:
                tmp = torch.mm(self.P[:i, :], self.P[i].reshape(-1, 1))
                self.P[i] -= torch.mm(self.P[:i, :].T, tmp).reshape(-1)
                tmp = torch.norm(self.P[i])
                self.P[i] /= tmp
        self.fitted = True

    def get_model_param_vec(self, model):
        """
        Return model parameters as a vector
        """
        vec = []
        for _, param in model.named_parameters():
            vec.append(param.reshape(-1))
        return torch.cat(vec, 0)    
    
    def get_model_grad_vec(self, model):
        """
        Return model grad as a vector
        """
        vec = []
        for _, param in model.named_parameters():
            vec.append(param.grad.reshape(-1))
        return torch.cat(vec, 0)
    
    def update_grad(self, model, grad_vec):
        """
        Update model grad
        """
        idx = 0
        for _, param in model.named_parameters():
            arr_shape = param.grad.shape
            size = arr_shape.numel()
            param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape).clone()
            idx += size
        