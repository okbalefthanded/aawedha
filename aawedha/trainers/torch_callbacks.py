'''
Base class for Pytorch Model CallBacks
'''
from copy import deepcopy


class CallBack:
    def __init__(self):
        self.epoch = 0
        self.model_state = {} 

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_test_begin(self):
        pass

    def on_test_end(self):
        pass

    def on_predict_begin(self):
        pass

    def on_predict_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_train_batch_begin(self):
        pass

    def on_train_batch_end(self):
        pass

class ModelCheckPoint(CallBack):

    def __init__(self, monitor="loss", mode="min", verbose=0):
        super().__init__()
        self.monitor = monitor
        self.mode    = mode
        self.model_state    = None
        self.tracked_metric = None 
        self.verbose    = verbose
        self.best_epoch = 0

    def on_train_begin(self, module):
        # self.model_state = module.state_dict()
        self.model_state = deepcopy(module.state_dict())

    def on_epoch_end(self, model, train_loader, epoch, val_metrics):
        
        if epoch == 0:
            self.tracked_metric = val_metrics[self.monitor]

        self.epoch = epoch      

        if self.mode == "min":
            if self.tracked_metric > val_metrics[self.monitor]:
                # self.model_state = model.module.state_dict()
                self.model_state = deepcopy(model.module.state_dict())
                self.tracked_metric = val_metrics[self.monitor]
                self.best_epoch = epoch
        else:
            if self.tracked_metric < val_metrics[self.monitor]:
                pass        

    def on_train_end(self, model, epoch):
        self.epoch = epoch
        model.module.load_state_dict(self.model_state)


