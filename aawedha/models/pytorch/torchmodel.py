from copy import deepcopy
from aawedha.models.pytorch.samtorch import enable_running_stats, disable_running_stats
from aawedha.models.pytorch.torch_builders import get_metrics, build_scheduler
from aawedha.evaluation.evaluation_utils import fit_scale, transform_scale
from aawedha.models.pytorch.torch_builders import get_optimizer, get_loss
from torchsummary import summary
import torch.nn as nn
import numpy as np
import collections
import torch
import pkbar


class TorchModel(nn.Module):

    def __init__(self, device='cuda', name='torchmodel'):
        super().__init__()
        self.optimizer = None
        self.loss = None
        self.metrics_list = []
        self.metrics_names = []
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
                scheduler=None):
        
        self.optimizer = get_optimizer(optimizer, self.parameters())
        self.loss = get_loss(loss)
        self.metrics_list = get_metrics(metrics)

        # transfer to device
        self.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics_list:
            metric.to(self.device)
        # 
        # self.optimizer_state = self.optimizer.state_dict()
        # self.optimizer_state = optimizer
        self.scheduler = scheduler

    def train_step(self, data):
        """
        """
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        if self._is_one_cycle():
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
        history, hist = {}, {}
        has_validation = False
        if validation_data:
            has_validation = True

        labels_type = self._labels_type()        

        if isinstance(x, torch.utils.data.DataLoader):
            train_loader = x
            self.input_shape, y_size = self._data_shapes(x)
            if y_size > 1:
                self._set_auroc_classes()
        else:
            self.input_shape = x.shape[1:]
            train_loader = self.make_loader(x, y, batch_size, shuffle=shuffle, 
                                            labels_type=labels_type) 
            if y.ndim > 1:
                self._set_auroc_classes()         

        self.set_output_shape()
        if self.is_categorical:
            self.is_binary = torch.tensor(y).shape[1] == 2
        else:
            self.is_binary = torch.tensor(y).unique().max() == 1       
        
        if class_weight: 
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    self.loss.pos_weight = torch.tensor([class_weight[0], class_weight[1]])        
        
        [metric.train() for metric in self.metrics_list]
        self.set_metrics_names()

        if self.scheduler:
            self.scheduler = build_scheduler(train_loader, self.optimizer, self.scheduler)

        if has_validation:
            if not isinstance(validation_data, torch.utils.data.DataLoader):
                validation_data = self.make_loader(validation_data[0], 
                                                   validation_data[1],
                                                   batch_size, shuffle,
                                                   labels_type)
        for metric in self.metrics_names:
            hist[metric] = []
            if has_validation:
                hist[f"val_{metric}"] = []
        
        if verbose == 2:
            progress = pkbar.Kbar(target=len(train_loader), width=25, always_stateful=True)       

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0
            self.reset_metrics()
            self.train()
            
            if verbose == 2:
                print("Epoch {}/{}".format(epoch+1, epochs))
            # train step
            for i, data in enumerate(train_loader, 0):
                return_metrics = self.train_step(data)
                running_loss += return_metrics['loss'] / len(train_loader)
                return_metrics['loss'] = running_loss

                if verbose == 2:
                    progress.update(i, values=[(k, return_metrics[k]) for k in return_metrics])
            
            # evaluate validation data
            val_metrics = None
            if has_validation:
                val_metrics = self.evaluate(validation_data, batch_size=batch_size, shuffle=False)
                for metric in val_metrics:
                    hist[f"val_{metric}"].append(val_metrics[metric])

            # update scheduler 
            if not self._is_one_cycle():
                self.update_scheduler()
            
            if verbose == 2:
                if has_validation:
                    progress.add(1, values=[(f"val_{k}", val_metrics[k]) for k in val_metrics])
                else:
                    progress.add(1)

            # update history
            for metric in return_metrics:
                hist[metric].append(return_metrics[metric])                
        
        history['history'] = hist
        return history

    def predict(self, x, normalize=False):
        """
        """
        if normalize:
            x = self.normalize(x)
        self.eval()
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(self.device)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        pred = self(x_tensor)
        #if self._is_binary():
        # if pred.ndim == 2:
        #    if pred.shape[1] == 1:
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
            test_loader = self.make_loader(x, y, batch_size, shuffle, labels_type)

        self.eval()
        self.loss.eval()        
        for metric in self.metrics_list:
            metric.eval()

        self.reset_metrics()

        if verbose == 2:
            progress = pkbar.Kbar(target=len(test_loader), width=25, always_stateful=True)
        
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running inputs through the network
                outputs = self(inputs)
                loss += self.loss(outputs, labels).item()

                # return_metrics = {'loss': loss.item() / len(test_loader)}
                return_metrics = {'loss': loss / len(test_loader)}
                return_metrics = self._compute_metrics(return_metrics, outputs, labels)

                if verbose == 2:
                    progress.update(i, values=[(k, return_metrics[k]) for k in return_metrics])
        
        if verbose == 2:
            progress.add(1)
        
        return return_metrics

    def set_metrics_names(self):
        """
        """
        if self.metrics_names:
            return
        else:
            self.metrics_names.append('loss')
            for metric in self.metrics_list:
                key = str(metric).lower()[:-2]
                if key == 'auroc':
                    key = 'auc'
                self.metrics_names.append(key)                

    def set_device(self, device=None):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_weights(self, state_dict):
        '''
        with torch.no_grad():
            for layer in self.state_dict():
                self.state_dict()[layer] = state_dict[layer]
        '''
        # TODO: for custom modules 
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # self.optimizer = get_optimizer(self.optimizer, self.parameters())
        # print(self.optimizer.state_dict())
        if self.optimizer:
            self.optimizer.state = collections.defaultdict(dict) # Reset state     
            # for st in self.optimizer_state:
            #    self.optimizer.state_dict()[st] = self.optimizer_state[st]
        
        # self.state_dict = state_dict

    def get_weights(self):
        return self.state_dict()

    def summary(self, shape=None):
        """
        """
        input_shape = None
        if shape:
            input_shape = shape
        elif self.input_shape:
            input_shape = self.input_shape
        if input_shape:
            summary(self, input_shape, device=self.device)

    def save(self, path):
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
        for metric in self.metrics_list:
            metric.reset()   
    
    def update_scheduler(self):
        if self.scheduler:
            self.scheduler.step()

    def set_output_shape(self):
        """Setter for output shape attribute
        """
        modules = list(self._modules.keys())
        output_index = -2
        if modules[-1] != 'loss':
            output_index = -1
        last_layer = modules[output_index]
        if hasattr(self._modules[last_layer], 'out_features'):
            self.output_shape = self._modules[last_layer].out_features

    def _compute_metrics(self, return_metrics, outputs, labels):
        with torch.no_grad():
            # if self._is_binary(labels):
            # torchmetrics requires sparse labels, some losses (eg polyLoss)
            # require categorical labels
            targets = deepcopy(labels)
            # if targets.shape[1] > 1:
            #   targets = targets.argmax(axis=1)
            if self.is_binary:
                outputs = nn.Sigmoid()(outputs)
                outputs = outputs.squeeze()
                targets = targets.squeeze()
                # hack for ECE
                if outputs.min() == 0:
                    outputs[outputs==0] += 1e-10
            for metric in self.metrics_list:
                metric_name = str(metric).lower()[:-2] 
                targets = self._labels_to_int(metric_name, targets)
                metric.update(outputs, targets)
                if metric_name == 'auroc':
                    metric_name = 'auc'
                return_metrics[metric_name] = metric.compute().item()
        return return_metrics
    
    # def _is_binary(self, labels):
    #     return labels.unique().max() == 1
        # return "BCE" in str(type(self.loss))
        
    def _is_one_cycle(self):
        return type(self.scheduler) is torch.optim.lr_scheduler.OneCycleLR
    
    def _set_auroc_classes(self):
        self.is_categorical = True
        # set AUROC num_classes to 2
        for metric in self.metrics_list:
            metric_name = str(metric).lower()[:-2]
            if metric_name == 'auroc':
                metric.num_classes = 2

    def _labels_to_int(self, metric, labels):
        if self.is_categorical and metric == 'auroc':
            return labels.argmax(axis=1).int()
        else:
            return labels.int()

    def _labels_type(self):
        labels_type = torch.long
        if type(self.loss) is nn.BCEWithLogitsLoss:
            labels_type = torch.float32
        return labels_type

    def _reshape_input(self, x):
        n, h, w = x.shape
        return x.reshape(n, 1, h, w)

    @staticmethod
    def _data_shapes(x):
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
    
    @staticmethod
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

    def initialize_glorot_uniform(self):
        for module in self.modules():
            if hasattr(module, 'weight'):
                if not("BatchNorm" in module.__class__.__name__):
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias"):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class SAMTorch(TorchModel):

    def train_step(self, data):
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        
        # first forward-backward step
        enable_running_stats(self)
        outputs = self(inputs)
                
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(self)
        self.loss(self(inputs), labels).backward()
        self.optimizer.second_step(zero_grad=True)
        
        return_metrics = {'loss': loss.item()}        
        return_metrics = self._compute_metrics(return_metrics, outputs, labels)
    
        return return_metrics