from aawedha.trainers.torchdata import make_loader
from aawedha.trainers.torchmodel import TorchModel
from torch.optim.swa_utils import SWALR
import aawedha.trainers.swa_utils as swa_utils
import torch.nn as nn
import torch
import pkbar


class SWA(TorchModel):
    def __init__(self, module=None, device='cuda', name='torchSWAmodel',
                 swa_start=50, swa_lr=0.05, anneal_epochs=50,
                 use_swa_scheduler=False):
        super().__init__(module=module, device=device, name=name)
        self.swa_model = None
        self.swa_scheduler = None
        if use_swa_scheduler:
            self.swa_scheduler = SWALR
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.anneal_epochs = anneal_epochs

    def compile(self, optimizer='Adam', loss=None, metrics=None,
                loss_weights=None, scheduler=None, classes=2):
        self._compile_regular(optimizer=optimizer, loss=loss, metrics=metrics,
                              scheduler=scheduler, classes=classes)

        if self.swa_scheduler:
            self.swa_scheduler = self.swa_scheduler(self.optimizer, swa_lr=self.swa_lr,
                                                    anneal_epochs=self.anneal_epochs,
                                                    anneal_strategy="cos")
        # is it the right spot to invoke this method?
        self.set_swa_model()

    def _fit_loop(self, train_loader, validation_data, has_validation,
                  epochs, batch_size, hist, progress, verbose):
        use_default = True
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0
            self.reset_metrics()
            self.module.train()

            if verbose == 2:
                print("Epoch {}/{}".format(epoch+1, epochs))

            # train step
            for i, data in enumerate(train_loader, 0):
                return_metrics = self.train_step(data)
                running_loss += return_metrics['loss'] / len(train_loader)
                return_metrics['loss'] = running_loss

                if verbose == 2:
                    progress.update(
                        i, values=[(k, return_metrics[k]) for k in return_metrics])
            if epoch > self.swa_start:
                use_default = False
                self.swa_model.update_parameters(self.module)
                if self.swa_scheduler:
                    self.swa_scheduler.step()
            else:
                # update scheduler
                if not self._cyclical_scheduler():
                    self.update_scheduler()

            # torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=self.device)
            swa_utils.update_bn(train_loader, self.swa_model, device=self.device)

            # evaluate validation data
            val_metrics = None
            if has_validation:
                val_metrics = self.evaluate(
                    validation_data, batch_size=batch_size, shuffle=False, use_default=use_default)
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
        # torch.optim.swa_utils.update_bn(train_loader, self.swa_model)
        return hist

    def predict(self, x, normalize=False, use_default=False):
        """
        """
        if normalize:
            x = self.normalize(x)
        self.module.eval()
        self.swa_model.eval()
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(self.device)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        if use_default:
            pred = self.module(x_tensor)
        else:
            pred = self.swa_model(x_tensor)

        if self.is_binary:
            pred = nn.Sigmoid()(pred)
        return pred.cpu().detach().numpy()

    def set_swa_model(self):
        # self.swa_model = torch.optim.swa_utils.AveragedModel(self.module)
        self.swa_model = swa_utils.AveragedModel(self.module)

    def evaluate(self, x, y=None, batch_size=32, verbose=0, normalize=False,
                 shuffle=False, return_dict=True, use_default=False):
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

        self.swa_model.eval()
        self.module.eval()
        self.loss.eval()
        [metric.eval() for metric in self.metrics_list]

        self.reset_metrics()

        if verbose == 2:
            progress = pkbar.Kbar(target=len(test_loader),
                                  width=25, always_stateful=True)

        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data[0].to(
                    self.device), data[1].to(self.device)
                # calculate outputs by running inputs through the network
                if use_default:
                    outputs = self.module(inputs)
                else:
                    outputs = self.swa_model(inputs)
                loss += self.loss(outputs, labels).item()

                return_metrics = {'loss': loss / len(test_loader)}
                return_metrics = self._compute_metrics(
                    return_metrics, outputs, labels)

                if verbose == 2:
                    progress.update(
                        i, values=[(k, return_metrics[k]) for k in return_metrics])

        if verbose == 2:
            progress.add(1)

        return return_metrics
