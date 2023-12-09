import numpy as np 
import torch

def data_shapes(x):
    if hasattr(x.dataset, 'tensors'):
        input_shape = x.dataset.tensors[0].shape[1:]                
        y_size = x.dataset.tensors[1].ndim
    else: 
        if hasattr(x.dataset, 'data'):
            input_shape = x.dataset.data.shape[1:]
        if isinstance(x.dataset.targets, list):
            y_size = np.array(x.dataset.targets).ndim
        else:
            y_size = x.dataset.targets.ndim
    return input_shape, y_size
    

def make_loader(x, y, batch_size=32, shuffle=True, labels_type=torch.long):
    """
    """
    if np.unique(y).size == 2 and y.ndim < 2:
        y = np.expand_dims(y, axis=1)
    tensor_set = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32), 
                                                torch.tensor(y, dtype=labels_type))
    loader = torch.utils.data.DataLoader(tensor_set, 
                                         batch_size=batch_size, 
                                         shuffle=shuffle)
    return loader

def reshape_input(x):
    """Reshape Torch Tensor from 3D to 4D:
    NHW -> N1HW

    Parameters
    ----------
    x : Torch Tensor
        input data to a model.

    Returns
    -------
    Torch Tensor
        input data to model.
    """
    n, h, w = x.shape
    return x.reshape(n, 1, h, w)