from aawedha.trainers.torchmodel import TorchModel
from aawedha.trainers.torchdata import make_loader
import torch
import pkbar

# Enorm
class Enorm(TorchModel):
    
    def __init__(self, module=None, device='cuda', name='torchmodel'):
        super().__init__()
        self.module = module
        self.optimizer = None
        self.loss = None
        self.loss_weights = None
        self.metrics_list = []
        self.metrics_names = []
        # self.callbacks = []
        self.scheduler = None
        # self.optimizer_state = None
        self.name = name
        self.history = {}
        self.device = device
        self.input_shape = None
        self.features_dim = None
        self.output_shape = None
        self.mu = None
        self.sigma = None  
        self.is_categorical = False 
        self.is_binary = False   
        self.enorm = None
    
    
    def train_step(self, data):
        """
        """
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.module(inputs)
        loss    = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.enorm.step()
        
        if self._cyclical_scheduler():
            self.update_scheduler()

        return_metrics = {'loss': loss.item()}
        return_metrics = self._compute_metrics(return_metrics, outputs, labels)
        return return_metrics