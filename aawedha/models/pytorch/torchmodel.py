from aawedha.models.pytorch.samtorch import enable_running_stats, disable_running_stats
from aawedha.evaluation.evaluation_utils import fit_scale, transform_scale
from torchsummary import summary
from ranger21 import Ranger21
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchmetrics
import torch
import pkbar

losses = {
    'binary_crossentropy': nn.BCEWithLogitsLoss,
    'sparse_categorical_crossentropy': nn.CrossEntropyLoss,
    'categorical_crossentropy': nn.CrossEntropyLoss
            }

available_metrics = {
    'accuracy': torchmetrics.Accuracy,
    'precision': torchmetrics.Precision,
    'recall': torchmetrics.Recall,
    'auc': torchmetrics.AUROC
    }


custom_opt = {
    'Ranger': Ranger21,
}


class TorchModel(nn.Module):

    def __init__(self, device='cuda', name='torchmodel'):
        super(TorchModel, self).__init__()
        self.optimizer = None
        self.loss = None
        self.metrics_list = []
        self.name = name
        self.history = {}
        self.device = device
        self.input_shape = None
        self.mu = None
        self.sigma = None  
        self.is_categorical = False      

    def compile(self, optimizer='Adam', loss=None,
                metrics=None, loss_weights=None):
        
        self.optimizer = self.get_optimizer(optimizer)
        self.loss = self.get_loss(loss)
        self.metrics_list = self.get_metrics(metrics)
        
        # transfer to device
        self.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics_list:
            metric.to(self.device)

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
        
        if isinstance(x, torch.utils.data.DataLoader):
            train_loader = x
            if hasattr(x.dataset, 'tensors'):
                self.input_shape = x.dataset.tensors[0].shape[1:]                
                y_size = x.dataset.tensors[1].ndim
            else: 
                if hasattr(x.dataset, 'data'):
                    self.input_shape = x.dataset.data.shape[1:]
                if isinstance(x.dataset.targets, list):
                    y_size = np.array(x.dataset.targets).ndim
                else:
                    y_size = x.dataset.targets.ndim

            if y_size > 1:
                self._set_auroc_classes()
        else:
            self.input_shape = x.shape[1:]
            train_loader = self.make_loader(x, y, batch_size, shuffle=shuffle) 
            if y.ndim > 1:
                self._set_auroc_classes()                
        
        if class_weight: 
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    self.loss.pos_weight = torch.tensor([class_weight[0], class_weight[1]])        

        hist['loss'] = []
        if validation_data:
            if not isinstance(validation_data, torch.utils.data.DataLoader):
                validation_data = self.make_loader(validation_data[0], 
                                                   validation_data[1], 
                                                   batch_size, shuffle)
            hist['val_loss'] = []

        for metric in self.metrics_list:
            key = str(metric).lower()[:-2]
            if key == 'auroc':
                key = 'auc'
            metric.train()
            hist[key] = []
            if validation_data:
                hist[f"val_{key}"] = []
        
        if verbose == 2:
            progress = pkbar.Kbar(target=len(train_loader), width=25, always_stateful=True)

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0
            self.reset_metrics()
            self.train()
            
            if verbose == 2:
                print("Epoch {}/{}".format(epoch+1, epochs))
                    
            for i, data in enumerate(train_loader, 0):                
                return_metrics = self.train_step(data)                
                running_loss += return_metrics['loss'] / len(train_loader)
                return_metrics['loss'] = running_loss

                if verbose == 2:
                    progress.update(i, values=[(k, return_metrics[k]) for k in return_metrics])
            
            # evaluate validation data
            val_metrics = None
            if validation_data:
                val_metrics = self.evaluate(validation_data, batch_size=batch_size, shuffle=False)
                for metric in val_metrics:
                    hist[f"val_{metric}"].append(val_metrics[metric])
            
            if verbose == 2:
                if val_metrics:
                    progress.add(1, values=[(f"val_{k}", val_metrics[k]) for k in val_metrics])
                else:
                    progress.add(1)

            # update history
            for metric in return_metrics:
                hist[metric].append(return_metrics[metric])                
        
        history['history'] = hist
        torch.cuda.empty_cache()
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
        if self._is_binary():
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
            test_loader = self.make_loader(x, y, batch_size, shuffle)

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
        torch.cuda.empty_cache()
        return return_metrics

    def get_optimizer(self, optimizer):
        """
        """
        params = {'params': self.parameters()}
        if isinstance(optimizer, str):
            return self._get_optim(optimizer, params)
        elif isinstance(optimizer, list):
            params = {**params, **optimizer[1]}
            return self._get_optim(optimizer[0], params)
        else:
            return optimizer

    @staticmethod
    def get_loss(loss):
        if loss in list(losses.keys()):
            return losses[loss]()
        else:
            raise ModuleNotFoundError

    @staticmethod
    def get_metrics(metrics):
        selected_metrics = []
        for metric in metrics:
            if isinstance(metric, str):
                selected_metrics.append(available_metrics[metric]())
            else:
                selected_metrics.append(metric)
        return selected_metrics

    def set_device(self, device=None):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_weights(self, state_dict):
        self.state_dict = state_dict

    def get_weights(self):
        return self.state_dict

    def summary(self):
        """
        """
        if self.input_shape:
            summary(self, self.input_shape, device=self.device)

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
    
    @staticmethod
    def _get_optim(opt_id, params):
        available = list(optim.__dict__.keys())
        if opt_id in available:
            return getattr(optim, opt_id)(**params)
        elif opt_id in custom_opt:
            return custom_opt[opt_id](**params)
        else:
            raise ModuleNotFoundError

    def _compute_metrics(self, return_metrics, outputs, labels):
        with torch.no_grad():
            for metric in self.metrics_list:
                metric_name = str(metric).lower()[:-2]
                if self._is_binary():
                    outputs = nn.Sigmoid()(outputs)
                labels = self._labels_to_int(metric_name, labels)                                    
                metric.update(outputs, labels)
                if metric_name == 'auroc':
                    metric_name = 'auc'
                return_metrics[metric_name] = metric.compute().item()
            return_metrics[metric_name] = metric.compute().item()
        
        return return_metrics
    
    def _is_binary(self):
        return "BCE" in str(type(self.loss))

    def _set_auroc_classes(self):
        self.is_categorical = True
        # set AUROC num_classes to 2
        for metric in self.metrics_list:
            metric_name = str(metric).lower()[:-2]
            if metric_name == 'auroc':
                metric.num_classes = 2

    def _labels_to_int(self, metric, labels):
        # if categorical
        if self.is_categorical and metric == 'auroc':
            return labels.argmax(axis=1).int()
        else:
            return labels.int()

    @staticmethod
    def make_loader(x, y, batch_size=32, shuffle=True):
        """
        """
        if np.unique(y).size == 2 and y.ndim < 2:
            y = np.expand_dims(y, axis=1)

        tensor_set = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32), 
                                                    torch.tensor(y))
        loader = torch.utils.data.DataLoader(tensor_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle)
        return loader


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