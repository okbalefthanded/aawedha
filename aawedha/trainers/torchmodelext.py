from aawedha.trainers.torchmodel import TorchModel
from aawedha.trainers.torchdata import make_loader
import torch
import pkbar

# TorchModelExtended
class TorchModelExt(TorchModel):
    
    def train_step(self, data):
        """
        """
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        # zero the parameter gradients
        for opt in self.optimizer:
            opt.zero_grad()

        # forward + backward + optimize
        outputs, features = self.module(inputs)
        total_loss = 0
        for loss, loss_weight in zip(self.loss, self.loss_weights):
            tensor = outputs if not self._loss_with_features(loss) else features
            # print(f"outputs {outputs.shape} features {features.shape} tensor {tensor.shape}")
            # print(f"lw {loss_weight} loss {loss(tensor, labels).shape}")
            # total_loss += loss_weight * loss(tensor, labels)
            total_loss = torch.add(total_loss, loss_weight * loss(tensor, labels))    
        total_loss.backward()
        for opt in self.optimizer:
            opt.step()
        
        if self._cyclical_scheduler():
            self.update_scheduler()

        return_metrics = {'loss': total_loss.item()}
        return_metrics = self._compute_metrics(return_metrics, outputs, labels)
        return return_metrics

    def evaluate(self, x, y=None, 
                 batch_size=32, 
                 verbose=0, 
                 normalize=False, 
                 shuffle=False, 
                 return_dict=True):
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
        [loss.eval() for loss in self.loss]
        if self.metrics_list:
            [metric.eval() for metric in self.metrics_list]        

        self.reset_metrics()

        if verbose == 2:
            progress = pkbar.Kbar(target=len(test_loader), width=25, always_stateful=True)
        
        with torch.no_grad():
            total_loss = 0
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running inputs through the network
                outputs, features = self.module(inputs)
                for loss, loss_weight in zip(self.loss, self.loss_weights):
                    tensor = outputs if not self._loss_with_features(loss) else features
                    total_loss = torch.add(total_loss, loss_weight * loss(tensor, labels))  

                return_metrics = {'loss': total_loss.item() / len(test_loader)}
                return_metrics = self._compute_metrics(return_metrics, outputs, labels)

                if verbose == 2:
                    progress.update(i, values=[(k, return_metrics[k]) for k in return_metrics])
        
        if verbose == 2:
            progress.add(1)
        
        return return_metrics

    @staticmethod
    def _loss_with_features(loss):
        return "uses_features" in loss.__dict__.keys()
        
    def calculate_total_loss(self, outputs, features, labels):
        total_loss = 0
        for loss, loss_weight in zip(self.loss, self.loss_weights):
            tensor = outputs
            if self._loss_with_features(loss):
                tensor = features
            total_loss += loss_weight * loss(tensor, labels)   
        return total_loss 