from aawedha.models.pytorch.torch_inits import initialize_Glorot_uniform
from aawedha.models.pytorch.samtorch import enable_running_stats
from aawedha.models.pytorch.samtorch import disable_running_stats
from aawedha.models.pytorch.torch_builders import build_callbacks
from aawedha.models.pytorch.torch_builders import build_scheduler
from aawedha.models.pytorch.torch_builders import get_optimizer
from aawedha.models.pytorch.torch_builders import get_metrics
from aawedha.models.pytorch.torch_builders import get_loss
from aawedha.evaluation.evaluation_utils import fit_scale
from aawedha.evaluation.evaluation_utils import transform_scale
from aawedha.models.pytorch.torchdata import data_shapes
from aawedha.models.pytorch.torchdata import make_loader
from torchsummary import summary
from copy import deepcopy
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
import numpy as np
import collections
import torch
import pkbar


class TorchModel(nn.Module):

    def __init__(self, module=None, device='cuda', name='torchmodel'):
        super().__init__()
        self.module = module
        self.optimizer = None
        self.loss = None
        self.metrics_list = []
        self.metrics_names = []
        self.callbacks = []
        self.scheduler = None
        # self.optimizer_state = None
        self.name = name
        self.history = {}
        self.device = device
        self.input_shape = None
        self.output_shape = None
        self.mu = None
        self.sigma = None  
        self.is_categorical = False 
        self.is_binary = False   

    def compile(self, optimizer='Adam', loss=None,
                metrics=None, loss_weights=None,
                scheduler=None, classes=2, callbacks=[]):
        self._compile_regular(optimizer=optimizer, loss=loss, metrics=metrics, 
                              scheduler=scheduler, classes=classes, callbacks=callbacks)

    def train_step(self, data):
        """
        """
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.module(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        if self._cyclical_scheduler():
            self.update_scheduler()

        return_metrics = {'loss': loss.item()}
        return_metrics = self._compute_metrics(return_metrics, outputs, labels)
        return return_metrics

    def fit(self, x, y=None, batch_size=32, epochs=100, verbose=2, 
            validation_data=None, class_weight=None, 
            steps_per_epoch=None, shuffle=True, 
            callbacks=None):        
        """
        """
        history, hist  = {}, {}
        train_loader, has_validation, validation_data = self._pre_fit(x, y, 
                                                                      batch_size, 
                                                                      validation_data, 
                                                                      class_weight, 
                                                                      shuffle)

        for metric in self.metrics_names:
            hist[metric] = []
            if has_validation:
                hist[f"val_{metric}"] = []
        
        progress = None
        if verbose == 2:
            progress = pkbar.Kbar(target=len(train_loader), width=25, always_stateful=True)

        hist = self._fit_loop(train_loader, validation_data, has_validation,
                              epochs, batch_size, hist, progress, verbose)              
        
        history['history'] = hist
        return history

    def _fit_loop(self, train_loader, validation_data, has_validation,
                 epochs, batch_size, hist, progress, verbose):
        # on_train_begin callbacks
        # if self.callbacks: self.callbacks.on_train_begin()
        if self.callbacks: [clbk.on_train_begin() for clbk in self.callbacks] 

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0
            self.reset_metrics()
            self.module.train()
            # on_epoch_begin callbacks
            if self.callbacks: [clbk.on_epoch_begin() for clbk in self.callbacks]
            if verbose == 2:
                print("Epoch {}/{}".format(epoch+1, epochs))
            
            # train step
            for i, data in enumerate(train_loader, 0):
                # on_train_batch_begin callbacks
                if self.callbacks: [clbk.on_train_batch_begin() for clbk in self.callbacks]
                return_metrics = self.train_step(data)
                running_loss  += return_metrics['loss'] / len(train_loader)
                return_metrics['loss'] = running_loss
                # on_train_batch_end callbacks
                if self.callbacks: [clbk.on_train_batch_end() for clbk in self.callbacks]
                if verbose == 2:
                    progress.update(i, values=[(k, return_metrics[k]) for k in return_metrics])
            
            # on_epoch_end callbacks
            if self.callbacks: [clbk.on_epoch_end(self, train_loader, epoch) for clbk in self.callbacks]

            # evaluate validation data
            val_metrics = None
            if has_validation:
                val_metrics = self.evaluate(validation_data, batch_size=batch_size, shuffle=False)
                for metric in val_metrics:
                    hist[f"val_{metric}"].append(val_metrics[metric])

            # update scheduler 
            if not self._cyclical_scheduler():
                self.update_scheduler()
            
            if verbose == 2:
                if has_validation:
                    progress.add(1, values=[(f"val_{k}", val_metrics[k]) for k in val_metrics])
                else:
                    progress.add(1)

            # update history
            for metric in return_metrics:
                hist[metric].append(return_metrics[metric]) 
        # on_train_end callbacks
        if self.callbacks: [clbk.on_train_end() for clbk in self.callbacks]
        return hist

    def predict(self, x, normalize=False):
        """
        """
        if normalize:
            x = self.normalize(x)
        self.module.eval()
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(self.device)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        pred = self.module(x_tensor)

        if self.is_binary:
            pred = nn.Sigmoid()(pred)
        return pred.cpu().detach().numpy()

    def evaluate(self, x, y=None, batch_size=32, verbose=0, normalize=False, 
                 shuffle=False, return_dict=True):
        """
        """
        loss = 0
        if normalize:
            x = self.normalize(x)
        if isinstance(x, torch.utils.data.DataLoader):
            test_loader = x
        else:
            labels_type = self._labels_type()
            test_loader = make_loader(x, y, batch_size, shuffle, labels_type)

        self.module.eval()
        self.loss.eval()
        [metric.eval() for metric in self.metrics_list]        

        self.reset_metrics()

        if verbose == 2:
            progress = pkbar.Kbar(target=len(test_loader), width=25, always_stateful=True)
        
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running inputs through the network
                outputs = self.module(inputs)
                loss += self.loss(outputs, labels).item()

                # return_metrics = {'loss': loss.item() / len(test_loader)}
                return_metrics = {'loss': loss / len(test_loader)}
                return_metrics = self._compute_metrics(return_metrics, outputs, labels)

                if verbose == 2:
                    progress.update(i, values=[(k, return_metrics[k]) for k in return_metrics])
        
        if verbose == 2:
            progress.add(1)
        
        return return_metrics

    def _pre_fit(self, x, y, batch_size, validation_data, class_weight, shuffle):
        has_validation = True if validation_data else False
        labels_type  = self._labels_type()
        train_loader = self._create_loader(x, y, shuffle, batch_size)         

        self.set_output_shape()
        self._set_is_binary(train_loader.dataset.tensors[1])
        
        if class_weight: 
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    self.loss.pos_weight = torch.tensor([class_weight[0], class_weight[1]])        
        
        [metric.train() for metric in self.metrics_list]

        if self.scheduler:
            self.scheduler = build_scheduler(train_loader, self.optimizer, self.scheduler)

        if has_validation:
            if not isinstance(validation_data, torch.utils.data.DataLoader):
                validation_data = make_loader(validation_data[0], 
                                              validation_data[1],
                                              batch_size, shuffle,
                                              labels_type)
        return train_loader, has_validation, validation_data

    def set_metrics_names(self, metrics):
        """
        """
        if self.metrics_names:
            return
        else:
            self.metrics_names.append('loss')
            for m in metrics:
                if isinstance(m, str):
                    self.metrics_names.append(m)
                else:
                    self.metrics_names.append(str(m).lower()[:-2])       

    def set_device(self, device=None):
        devices = ['cuda', 'cpu']
        if device:
            if device not in devices:
                raise NotImplementedError
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_weights(self):
        """Default weights and bias initialization scheme.
        following Glorot uniform method.
        """
        initialize_Glorot_uniform(self.module)
    
    def set_weights(self, state_dict):
        '''
        with torch.no_grad():
            for layer in self.state_dict():
                self.state_dict()[layer] = state_dict[layer]
        '''
        # TODO: for custom modules
        if hasattr(self, 'init_weight'):
            self.init_weights()
        else:
            for layer in self.module.children():
                if isinstance(layer, nn.Sequential):
                    [l.reset_parameters() for l in layer if hasattr(l, 'reset_parameters')]
                if hasattr(layer, 'module'):
                    # layers that are composed of modules
                    layer.module.reset_parameters()
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        if self.optimizer:
            self.optimizer.state = collections.defaultdict(dict) # Reset state    

    def get_weights(self):
        return self.module.state_dict()

    def summary(self, shape=None):
        """Display the model layers and their input/output shapes.       

        Parameters
        ----------
        shape : tuple, optional
            data input shape (channels x samples), by default None
        """
        input_shape = None
        if shape:
            input_shape = shape
        elif self.input_shape:
            input_shape = self.input_shape
        if input_shape:
            summary(self.module, input_shape, device=self.device)

    def save(self, path):
        """Save model

        Parameters
        ----------
        path : str
            saved model path
        """
        torch.save(self, path)  

    def set_scale(self, x):
        """
        """
        x, self.mu, self.sigma = fit_scale(x)
        return x

    def normalize(self, x):
        """
        """
        return transform_scale(x, self.mu, self.sigma)

    def reset_metrics(self):
        """Reset metric values
        """
        for metric in self.metrics_list:
            metric.reset()   
    
    def update_scheduler(self):
        if self.scheduler:
            self.scheduler.step()

    def set_output_shape(self):
        """Setter for output shape attribute
        """
        modules = list(self.module._modules.keys())
        output_index = -2
        if modules[-1] != 'loss':
            output_index = -1
        last_layer = modules[output_index]
        if hasattr(self.module._modules[last_layer], 'out_features'):
            self.output_shape = self.module._modules[last_layer].out_features
        else:
            # Linear with TorchLayers regularizes
            self.output_shape = self.module._modules[last_layer].module.out_features

    def metrics_to(self, device=None):
        if not device:
            device = self.device
        for metric in self.metrics_list:
            metric.to(device)        
    
    def _compile_regular(self, optimizer='Adam', loss=None,
                metrics=None, loss_weights=None,
                scheduler=None, classes=2, callbacks=[]):
        self.optimizer = get_optimizer(optimizer, self.module.parameters())
        self.loss      = get_loss(loss)
        self.metrics_list = get_metrics(metrics, classes)
        self.callbacks = build_callbacks(self, callbacks)
        self.scheduler = scheduler
        self.set_metrics_names(metrics)
        # transfer to device
        self._to_device()

    def _to_device(self):
        """Transfer module, loss and metrics to compute device.
        """
        self.module.to(self.device)
        self.loss.to(self.device)
        [metric.to(self.device) for metric in self.metrics_list]    
    
    def _compute_metrics(self, return_metrics, outputs, labels):
        with torch.no_grad():
            # if self._is_binary(labels):
            # torchmetrics requires sparse labels, some losses (eg polyLoss)
            # require categorical labels
            targets = deepcopy(labels)
            if self.is_binary:
                outputs = nn.Sigmoid()(outputs)
                outputs = outputs.squeeze()
                targets = targets.squeeze()
                # hack for ECE
                if outputs.min() == 0:
                    outputs[outputs==0] += 1e-10
            for i, metric in enumerate(self.metrics_list):
                # metric_name = str(metric).lower()[:-2] 
                metric_name = self.metrics_names[i+1]
                targets = self._labels_to_int(metric_name, targets)
                metric.update(outputs, targets)
                return_metrics[metric_name] = metric.compute().item()
        return return_metrics

    def _set_is_binary(self, loader):
        if isinstance(loader, torch.Tensor):
            y = loader
        else:
            if hasattr(loader.dataset, 'tensors'):
                y = loader.dataset.tensors[1]
            elif hasattr(loader.dataset, 'data'):
                y = loader.dataset.targets            
        if self.is_categorical:
            self.is_binary = y.shape[1] == 2
        else:
            self.is_binary = y.unique().max() == 1        
        
    def _cyclical_scheduler(self):
        schedulers = [lrs.OneCycleLR, lrs.CyclicLR, lrs.CosineAnnealingWarmRestarts]
        return type(self.scheduler) in schedulers
    
    def _set_auroc_classes(self):
        self.is_categorical = True

    def _labels_to_int(self, metric, labels):
        if self.is_categorical and metric == 'auc':
            return labels.argmax(axis=1).int()
        else:
            return labels.int()

    def _labels_type(self):
        labels_type = torch.long
        if type(self.loss) is nn.BCEWithLogitsLoss:
            labels_type = torch.float32
        return labels_type

    def _create_loader(self, x, y, shuffle=True, batch_size=32):
        labels_type = self._labels_type()  
        
        if isinstance(x, torch.utils.data.DataLoader):
            train_loader = x
            self.input_shape, y_size = data_shapes(x)
            if y_size > 1:
                self._set_auroc_classes()
        else:
            self.input_shape = x.shape[1:]
            train_loader = make_loader(x, y, batch_size, shuffle=shuffle, 
                                            labels_type=labels_type) 
            if y.ndim > 1:
                self._set_auroc_classes()
        
        return train_loader    

class SAMTorch(TorchModel):

    def train_step(self, data):
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        
        # first forward-backward step
        enable_running_stats(self)
        outputs = self.module(inputs)
                
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(self)
        self.loss(self.module(inputs), labels).backward()
        self.optimizer.second_step(zero_grad=True)
        
        return_metrics = {'loss': loss.item()}
        return_metrics = self._compute_metrics(return_metrics, outputs, labels)
    
        return return_metrics