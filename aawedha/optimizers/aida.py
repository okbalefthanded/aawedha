# Taken from official code repo : https://github.com/guoqiang-x-zhang/AidaOptimizer
# Implements Aida optimizer published in:
# Zhang G, Niwa K, Kleijn WB. A DNN Optimizer that Improves over AdaBelief by 
# Suppression of the Adaptive Stepsize Range. 2022 Mar 24;(4). 
# Available from: http://arxiv.org/abs/2203.13273
 
import math
import torch
from torch.optim.optimizer import Optimizer

version_higher = ( torch.__version__ >= "1.5.0" )

class Aida(Optimizer):
    r"""Implements Aida algorithm. Modified from AdaBelief in PyTorch

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        K (int, optiional): number of vector projected per iteration
        xi (float, optional): term used in vector projections to avoid 
            division by zero (default: 1e-20)

    reference: G. Zhang, K. Niwa, and W. B. Kleijn, "On Exploiting Layerwise Gradient Statistics 
               for Effective Training of Deep Neural Networks", arXiv:2203.13273v2, March, 2022
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, K=2, xi=1e-20):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.K = K
        self.xi=xi 
        super(Aida, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Aida, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data,
                                   memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data,
                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
               
                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data,
                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data,
                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # perform weight decay
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Update first moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # Update second moment running average
                proj_g = grad.detach().clone()
                proj_m = exp_avg.detach().clone()
                for index in range(self.K):
                    scalar_g = torch.sum(torch.mul(proj_g, proj_m))/(torch.sum(torch.pow(proj_g,2))+self.xi)
                    scalar_m = torch.sum(torch.mul(proj_g, proj_m))/(torch.sum(torch.pow(proj_m,2))+self.xi)
                    proj_g.mul_(scalar_g)
                    proj_m.mul_(scalar_m)

                
                grad_residual =   proj_m - proj_g
                exp_avg_var.mul_(beta2).add_(1 - beta2, torch.pow(grad_residual,2))


                denom = ((exp_avg_var.add_(group['eps'])).sqrt() / math.sqrt(bias_correction2))#.add_(group['eps'])

                #update model     
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
                
        return loss