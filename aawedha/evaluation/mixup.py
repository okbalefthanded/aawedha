# mixup[1] augmentation technique implementation following Keras.io example
# https://keras.io/examples/vision/mixup/
# [1] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond Empirical Risk Minimization,
#  1–11. Retrieved from http://arxiv.org/abs/1710.09412
from aawedha.evaluation.evaluation_utils import labels_to_categorical
from torch.utils.data import TensorDataset
import numpy as np    
import random
import torch

def build_mixup_dataset(X_train, Y_train, X_val, Y_val, aug, batch):
    """Create a TF dataset instance with mixup lambda function. all data labels must be
    converted to categorical before fitting the model.

    Parameters
    ----------
    X_train : ndarray (N_training, channels, samples)
            training data
    X_val : ndarray (N_validation, channels, samples)
            validation data
    X_test : ndarray (N_test, channels, samples)
            test data
    Y_train : 1d array
            training data labels
    Y_val : 1d array
            validation data labels
    Y_test : 1d array 
            test data labels
    aug : str or list
        data augmentation method name is str.
        if list: first elements is the method name and the 2nd holds its parameter.
        eg: for mixup: ["mixup", 0.1], 0.1 is the value of alpha.
    batch : int
        batch size

    Returns
    -------
    tuple: 
        Pytorch: DataLoder instance.
    X_train
        training data with augmented mixup method.    
    val
        validation data tuple, with X_val dataset and Y_val categorical
        labels.        
    """
    alpha = 0.2
    if isinstance(aug, list):
        alpha = aug[1]
    return mixup_torch(X_train, Y_train, X_val, Y_val, alpha, batch)  
    
def mixup_torch(X_train, Y_train, X_val, Y_val, alpha, batch):
    
    Y_train = labels_to_categorical(Y_train)
    val = None
    tensor_set = MixUpDataSet(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train), alpha=alpha)
    X_train_loader = torch.utils.data.DataLoader(tensor_set, 
                                             batch_size=batch, 
                                             shuffle=True)
    if isinstance(Y_val, np.ndarray):
        Y_val = labels_to_categorical(Y_val)
        val_set = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
        val = torch.utils.data.DataLoader(val_set, 
                                             batch_size=batch, 
                                             shuffle=False)
    return X_train_loader, val


class MixUpDataSet(TensorDataset):

    def __init__(self, *tensors, alpha=0.1):
        super(MixUpDataSet, self).__init__(*tensors)
        self.alpha = alpha

    def __getitem__(self, index):
        tensor = self.tensors[0][index]
        label = self.tensors[1][index] 
        
        mixup_idx = random.randint(0, len(self.tensors[0])-1)
        mixup_tensor = self.tensors[0][mixup_idx]
        mixup_label = self.tensors[1][mixup_idx] 

        lam = np.random.beta(self.alpha, self.alpha)
        
        tensor = lam * tensor + (1 - lam) * mixup_tensor
        label = lam * label + (1 - lam) * mixup_label

        return (tensor, label)
        