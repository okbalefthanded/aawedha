# Implements ADCL optimizer from:
# Zhang, Z., He, Y., Mai, W., Luo, Y., Li, X., Cheng, Y., Huang, X. and Lin, R. (2025) 
# ‘Convolutional Dynamically Convergent Differential Neural Network for Brain Signal Classification’, 
# IEEE Transactions on Neural Networks and Learning Systems, 36(5), pp. 8166–8177. 
# Available at: https://doi.org/10.1109/TNNLS.2024.3437676.
# Adapted from official implementation: https://github.com/BIRLab/ConvDCDNN

import torch


class ADCL(torch.optim.Optimizer):
    """
    Neural Network Optimizer Based on Neural Dynamic

    Examples
    --------

    Define neural network and loss function

    >>> model: torch.nn.Module
    >>> loss_function: torch.nn.Module

    Create an ADCL optimizer

    >>> optimizer = ADCL(model.parameters())

    Define a closure

    >>> x: torch.Tensor
    >>> y: torch.Tensor
    >>> def closure():
    ...     optimizer.zero_grad()
    ...     _pred = model(x)
    ...     if torch.isinf(_pred).any():
    ...         raise ValueError('Training loss is diverging, please decrease the learning rate.')
    ...     _loss = loss_function(_pred, y)
    ...     _loss.backward()
    ...     return _loss

    Optimization step

    >>> loss = optimizer.step(closure)

    Now you can use ADCL just like any other optimizer in PyTorch!

    """

    def __init__(
        self,
        params,
        lr=1e-3,
        activation_fn=None,
        vlr_clamp=1.0,
        weight_decay=1e-5,
        trapezoidal=False,
        omicron=1e-6
    ):
        r"""
        Initialize the ADCL optimizer.

        :param params: learnable parameters
        :param lr: time constant, similar to learning rate - :math:`\lambda^*`
        :param activation_fn: activation function - :math:`\Phi(\cdot)`
        :param vlr_clamp: maximum equivalent learning rate - :math:`\alpha_{\text{max}}`
        :param weight_decay: weight decay (L2 penalty) - :math:`\mu`
        :param trapezoidal: whether to use trapezoidal integration
        :param omicron: infinitesimal float used to ensure numerical stability - :math:`o`
        """
        super().__init__(params, defaults={
            'lr': lr,
            'activation_fn': activation_fn,
            'vlr_clamp': vlr_clamp,
            'weight_decay': weight_decay,
            'trapezoidal': trapezoidal,
            'omicron': omicron
        })

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError('ADCL optimizer need to pass loss through closure, please reference: "https://pytorch.org/docs/stable/optim.html#optimizer-step-closure".')

        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # calculate varying learning rate
                    if group['activation_fn'] is None:
                        vlr = group['lr'] * loss / (torch.sum(torch.square(p.grad)) + group['omicron'])
                    else:
                        vlr = group['lr'] * group['activation_fn'](loss) / (torch.sum(torch.square(p.grad)) + group['omicron'])

                    # learning rate clamp
                    if vlr > group['vlr_clamp']:
                        vlr = group['vlr_clamp']

                    # l2 regularization
                    if group['weight_decay'] > 0:
                        p.grad = p.grad.add(p, alpha=group['weight_decay'])

                    # optimization step
                    current_step = vlr * p.grad
                    if 'last_step' in self.state[p]:
                        p.data -= 0.5 * (current_step + self.state[p]['last_step'])
                    else:
                        p.data -= current_step

                    # trapezoidal integration
                    if group['trapezoidal']:
                        self.state[p]['last_step'] = current_step

        return loss


__all__ = ['ADCL']
