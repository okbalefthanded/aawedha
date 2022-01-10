from aawedha.utils.evaluation_utils import fit_scale, transform_scale
from torchsummary import summary 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchmetrics
import torch
import pkbar


class TorchModel(nn.Module):

    def __init__(self, device='cuda', name='torchmodel'):
        super(TorchModel, self).__init__()
        self.optimizer = None
        self.loss = None
        self.metrics_list = []
        self.name = ''
        self.history = {}
        self.device = device
        self.input_shape = None
        self.mu = None
        self.sigma = None        

    def compile(self, optimizer='Adam', loss=None,
                metrics=None, loss_weights=None):
        
        self.optimizer = self.get_optimizer(optimizer)
        self.loss = self.get_loss(loss)
        self.metrics_list = self.get_metrics(metrics)

    def fit(self, x, y, batch_size=32, epochs=100, 
            verbose=2, validation_data=None, 
            class_weight=None, steps_per_epoch=None, callbacks=None):        
        """
        """
        history, hist = {}, {}
        self.input_shape = x.shape[1:]

        train_loader = self.make_loader(x, y, batch_size)
        
        if class_weight:
            self.loss.pos_weight = torch.tensor(class_weight[1])
        
        self.to(self.device)
        self.loss.to(self.device)
        self.train()
        hist['loss'] = []
        if validation_data:
          hist['val_loss'] = []
        for metric in self.metrics_list:
            key = str(metric).lower()[:-2]
            metric.to(self.device)
            metric.reset()
            metric.train()
            hist[key] = []
            if validation_data:
                hist[f"val_{key}"] = []
        total_loss = 0
        progress = pkbar.Kbar(target=len(train_loader), width=25)
        for epoch in range(epochs):  # loop over the dataset multiple times            
            if verbose == 2:
                print("Epoch {}/{}".format(epoch+1, epochs))
                    
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item() / len(train_loader)                
                loss.backward()
                self.optimizer.step()
                
                # return_metrics = {'loss': loss.item()}
                return_metrics = {'loss': total_loss}
                for metric in self.metrics_list:                    
                    metric.update(torch.nn.Sigmoid()(outputs.to(self.device)), labels.int())                  
                    return_metrics[str(metric).lower()[:-2]] = metric.compute().item()
                
                if verbose == 2:
                    progress.update(i, values=[(k,return_metrics[k]) for k in return_metrics])
            
            # evaluate validation data
            val_metrics = None
            if validation_data:
                val_metrics = self.evaluate(validation_data[0], validation_data[1])
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

    def predict(self, x):
        # return self.model(torch.tensor(x).to(self.device))
        self.eval()
        pred = self(torch.tensor(x).to(self.device))
        if self._is_binary():
          pred = nn.Sigmoid()(pred)   
        # return self(torch.tensor(x).to(self.device)).cpu().detach().numpy()
        return pred.cpu().detach().numpy()

    def evaluate(self, x, y, batch_size=32, verbose=0):
        """
        """
        loss = 0
        test_loader = self.make_loader(x, y, batch_size)
        self.to(self.device)
        self.eval()
        
        for metric in self.metrics_list:
            metric.to(self.device)
            metric.eval()
            metric.reset()        
        
        progress = pkbar.Kbar(target=len(test_loader), width=25)

        for i, data in enumerate(test_loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            # calculate outputs by running inputs through the network
            outputs = self(inputs)
            outputs = outputs.to(self.device)
            loss += self.loss(outputs, labels)
            
            # metric.update(outputs.to(device), labels.to(device))
            return_metrics = {'loss': loss.item() / len(test_loader)}
            
            
            for metric in self.metrics_list:                
                metric.update(outputs, labels.int())                    
                return_metrics[str(metric).lower()[:-2]] = metric.compute().item()
            if verbose == 2:
                progress.update(i, values=[(k, return_metrics[k]) for k in return_metrics])
        
        # return_metrics = {'loss': loss / len(test_loader)}
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

    def get_loss(self, loss):
        losses = {'binary_crossentropy': nn.BCEWithLogitsLoss,
                   'sparse_categorical_crossentropy': nn.CrossEntropyLoss,
                   'categorical_crossentropy': nn.CrossEntropyLoss
                  }
        if loss in list(losses.keys()):
            return losses[loss]()
        else:
            raise ModuleNotFoundError

    def get_metrics(self, metrics):
        selected_metrics = []
        available_metrics = {'accuracy': torchmetrics.Accuracy,
                     'precision': torchmetrics.Precision,
                     'recall': torchmetrics.Recall,
                     'auc': torchmetrics.AUROC}
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

    def _get_optim(self, opt_id, params):
        available = list(optim.__dict__.keys())
        if opt_id in available:
            return getattr(optim, opt_id)(**params)
        else:
            raise ModuleNotFoundError

    def _is_binary(self):
      return "BCE" in str(type(self.loss))    

    @staticmethod
    def make_loader(x, y, batch_size=32):
        """
        """
        if np.unique(y).size == 2:   
          y = np.expand_dims(y, axis=1)

        tensor_set = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32), 
                                               torch.tensor(y))
        loader = torch.utils.data.DataLoader(tensor_set, 
                                             batch_size=batch_size, 
                                             shuffle=False)
        return loader