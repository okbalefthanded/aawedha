from aawedha.models.pytorch.torchmodel import TorchModel
from aawedha.models.pytorch.twa import TWA
import torch


class TwaTrain(TorchModel):

    def __init__(self, module=None, device='cuda', name='torchTWAmodel',
                twa_start=0, twa_end=100):
        super().__init__(module=module, device=device, name=name)
        self.twa = TWA(device=device)
        self.twa_start = twa_start
        self.twa_end =  twa_end   
        self.initial_lr = 0.  
        
    def train_step(self, data, epoch):
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
        if epoch > self.twa_end:
            # self.twa_model.update_parameters(self.module)
            self.twa.update_parameters(self.module)
        self.optimizer.step()
        
        if self._cyclical_scheduler():
            self.update_scheduler()

        return_metrics = {'loss': loss.item()}
        return_metrics = self._compute_metrics(return_metrics, outputs, labels)
        return return_metrics      
    
    def _fit_loop(self, train_loader, validation_data, has_validation,
                  epochs, batch_size, hist, progress, verbose):        
        self.initial_lr = self.optimizer.param_groups[-1]['lr']
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0
            self.reset_metrics()
            self.module.train()

            if verbose == 2:
                print("Epoch {}/{}".format(epoch+1, epochs))

            # train step
            for i, data in enumerate(train_loader, 0):
                return_metrics = self.train_step(data, epoch)
                running_loss += return_metrics['loss'] / len(train_loader)
                return_metrics['loss'] = running_loss

                if verbose == 2:
                    progress.update(
                        i, values=[(k, return_metrics[k]) for k in return_metrics])
            if epoch > self.twa_end:
                # self.twa_model.update_parameters(self.module)
                torch.optim.swa_utils.update_bn(train_loader, self.module, device=self.device)
            else:
                # update scheduler
                if not self._cyclical_scheduler():
                    self.update_scheduler()

            # evaluate validation data
            val_metrics = None
            if has_validation:
                # val_metrics = self.evaluate(validation_data, batch_size=batch_size, shuffle=False, use_default=use_default)
                val_metrics = self.evaluate(validation_data, batch_size=batch_size, shuffle=False)
                for metric in val_metrics:
                    hist[f"val_{metric}"].append(val_metrics[metric])

            if verbose == 2:
                if has_validation:
                    progress.add(
                        1, values=[(f"val_{k}", val_metrics[k]) for k in val_metrics])
                else:
                    progress.add(1)
                    
            # update history
            for metric in return_metrics:
                hist[metric].append(return_metrics[metric])
        
            if epoch == self.twa_end:
                self.twa.fit_subspace()
                self.init_lr()           
            
            if epoch >= self.twa_start and epoch < self.twa_end:
                self.twa.collect_solutions(self.module)
        
        return hist
        
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
        
    