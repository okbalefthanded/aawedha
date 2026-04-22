'''
Base class for Pytorch Model CallBacks
'''
from copy import deepcopy
from pathlib import Path
import warnings
import torch 


class CallBack:
    def __init__(self):
        self.epoch = 0
        self.model_state = {} 
        self.disable = False

    def on_train_begin(self, **kwargs):
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

    def disable_callback(self, epoch, val_metrics):
        if epoch == 0 and not val_metrics:
            warnings.warn(f"Absence of validation data. {self.__class__.__name__} will be disabled.")
            self.disable = True
            return True    
        return False

class ModelCheckPoint(CallBack):

    def __init__(self, monitor="loss", mode="min", verbose=0):
        super().__init__()
        self.monitor = monitor
        self.mode    = mode
        self.model_state    = None
        self.tracked_metric = None 
        self.verbose    = verbose
        self.best_epoch = 0

    def on_train_begin(self, model, **kwargs):
        # self.model_state = module.state_dict()
        self.model_state = deepcopy(model.state_dict())

    def on_epoch_end(self, model, train_loader, epoch, val_metrics):
               
        if self.disable_callback(epoch, val_metrics):
                return
        else:
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


class EarlyStopping(CallBack):
    """
    Keras-style Early Stopping for PyTorch.
    """
    def __init__(self, monitor="loss", mode="min", patience=7, 
                 min_delta=0, verbose=False, path='checkpoint.pt', 
                 restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = Path(path)
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss = None
        self.stop_training = False
        self.best_weights = None

    def on_epoch_end(self, model, train_loader, epoch, val_metrics):
        
        if self.disable_callback(epoch, val_metrics):
            return

        if self.best_loss is None:
            self.best_loss = val_metrics[self.monitor]
            self.save_checkpoint(model)
        elif val_metrics[self.monitor] > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose:
                        print("Restoring best model weights.")
                    model.load_state_dict(self.best_weights)
        else:
            self.best_loss = val_metrics[self.monitor]
            self.save_checkpoint(model)
            self.counter = 0

    def on_train_end(self, model, epoch):
        return super().on_train_end()

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased. Saving model...')
        
        # Save to disk
        torch.save(model.state_dict(), self.path)
        
        # Save to memory for quick restoration
        if self.restore_best_weights:
            self.best_weights = deepcopy(model.state_dict())

