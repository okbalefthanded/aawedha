from aawedha.evaluation.evaluation_utils import metrics_binary
from aawedha.trainers.builder_model import build_learner
import torch

class Learner:
    """Wrapper class to unify training and inference operations for different
    Frameworks.
    """
    def __init__(self, 
                 model=None, 
                 compiled=False, 
                 weights={},
                 config={}, 
                 history=[], 
                 normalize=True,
                 name=None):
        """Constructor
        initiliaze Learner object with empty attributes.
        The object update will be made at compile method call.

        Parameters
        ----------
        model :  TorchModel, optional
            the model instance that will be trained, by default None
        compiled : bool, optional
            True if the model is compiled, by default False
        weights : dict, optional
            the model's weights at creation, used to keep the same start 
            point in later evaluations.
            - dict : TorchModel module initial state.
            by default {}
        config : dict, optional
            compile and training configuration, by default {}
        history : a list of : dict, optional
            Record of training/validation metrics and loss values per epoch, by default []
        normalize : bool, optional
            if True, normalize inputs before training and evaluation, by default True
        name : str, optional
            model name, by default None
        """
        self.model    = model        
        self.compiled = compiled        
        self.config   = config
        self.history  = history
        self.name = name
        self.initial_weights = weights
        self.do_normalize    = normalize        

    def set_model(self, model):
        """model setter 
        - Pytorch : create a TorchModel (or subclass) object, then assign model to
        the TorchModel module attribute.

        Parameters
        ----------
        model : Pytorch nn Module
            model object for training and evalution.
        """
        if not self.model:
            self.model = build_learner(self.config['compile'])
            self.model.module = model
            self.model.name   = model.name
        self.name = model.name

    def compile(self, device, classes):
        """Compile model: create training configuration objects
        from description

        Parameters
        ----------
        device : str
            compute hardware: {CPU | GPU}
        classes : int
            number of classes in the dataset to be trained on
        """
        devices = {'CPU': 'cpu', 'GPU': 'cuda'}
        khsara, optimizer, metrics, loss_weights, schedule = self.get_compile_configs(device, classes)
        self.model.set_device(devices[device.upper()])
        self._compile(khsara, optimizer, metrics, loss_weights, schedule, classes)
        self.compiled = True

    def fit(self, x, y, 
            batch_size, 
            epochs,
            verbose, 
            validation_data, 
            class_weight, 
            callbacks):
        """Trains the model with a similar Keras fit method.

        Parameters
        ----------
        x : {Numpy Array, PyTorch DataLoader} 
            Training data
        y : {Numpy Array, None}
            Label data.
            (None in case of x is PyTorch DataLoader)
        batch_size : int
            number of samples per one training step
        epochs : int
            number of training epochs.
        verbose : {0, 1, 2}
            print training progress per step.
            0 : silent
            1 : progress bar
            2 : one line per epoch
        validation_data : {None, Numpy Array, Pytorch Loader}
            data to test model on after each epoch end.
            if None, no validation.
        class_weight : dict
            class weights mapping.
        callbacks : list of callback objects
            methods to be called only during training.

        Returns
        -------
        history : dict, optional
            Record of training/validation metrics and loss values per epoch, by default []
            - dict: if the model is a TorchModel object.
        """
        return self.model.fit(x=x, y=y, 
                              batch_size=batch_size,
                              epochs=epochs, 
                              verbose=verbose, 
                              validation_data=validation_data,
                              class_weight=class_weight, 
                              callbacks=callbacks)

    def predict(self, X):
        """Generate output predictions for the input samples.
        Parameters
        ----------
        X : ndarray
            input data samples x channels 
        Returns
        -------
        preds : ndarray
            model predictionss
        """
        preds = self.model.predict(X, normalize=self.do_normalize)
        return preds

    def evaluate(self, 
                 X, 
                 Y, 
                 batch_size=32, 
                 return_dict=True, 
                 verbose=0):
        """_summary_

        Parameters
        ----------
        X : {Numpy Array, PyTorch DataLoader} 
            Training data
        Y : {Numpy Array, None}
            Label data.
            (None in case of X is PyTorch DataLoader)
        batch_size : int
            number of samples per one training step
        return_dict : bool, optional (Keras legacy argument)
            return results as a dict, by default True 
        verbose : {0, 1, 2}
            print training progress per step.
            0 : silent
            1 : progress bar
            2 : one line per epoch

        Returns
        -------
        dict
            a dictionary of metrics results in the form {metric: value}
        """
        perfs = {}
        perfs = self.model.evaluate(X, Y, 
                                     batch_size=batch_size, 
                                     normalize=self.do_normalize,
                                     return_dict=return_dict, 
                                     verbose=verbose)
        return perfs
    
    def save(self, filepath):
        """Save the TorchModel using its inner saving method save()

        Parameters
        ----------
        filepath : str
            path to store the model
        """
        self.model.save(filepath)

    def reset_weights(self):
        """Reset model weights to initial state
        Uses the TorchModel set_weight() method
        """
        # if not self.initial_weights: 
        #     self.initial_weights['model_weights'] = self.model.get_weights()
        # else:
        #     self.model.set_weights(self.initial_weights['model_weights'])
        self.model.set_weights()

    def get_compile_configs(self, device, classes):
        """Returns default model compile configurations as tuple

        Parameters
        ----------
        device : str
            compute hardware: {CPU | GPU}
        classes : int
            number of classes in the dataset to be trained on

        Returns
        -------
        khsara (loss in Arabic): str
            loss function optimized during training
        optimizer : str | dict
            optimizer congfiguration to be built later
        metrics : list : str 
            metrics measuring the model's performance in evaluaion
        loss_weights: list, (n_classes)
            weight of each classes when using a combination of losses, None default 
        schedule: str | dict
            learning rate schedule configuration 
        """ 
        schedule, loss_weights = None, None
        if 'compile' in self.config:
            # metrics
            if 'metrics' in self.config['compile']:
                metrics = self.config['compile']['metrics']
            else:
                metrics = self._get_metrics(classes)
            # loss (es)
            if 'loss' in self.config['compile']:
                khsara = self.config['compile']['loss']
            else:
                khsara = self._get_loss(classes)
            # optimizer (s)
            if 'optimizer' in self.config['compile']:
                optimizer = self.config['compile']['optimizer']
            else:
                optimizer = 'adam'
            # loss weights            
            if 'loss_weights' in self.config['compile']:
                loss_weights = self.config['compile']['loss_weights']
            # scheduler
            if 'scheduler' in self.config['compile']:
                schedule = self.config['compile']['scheduler']
        else:
            khsara, optimizer, metrics = self._default_compile(device, classes)           

        return khsara, optimizer, metrics, loss_weights, schedule

    def fit_normalize(self, X_train):
        """Adapt normalization layer if it's inside the model
        Parameters
        ----------
        X_train : ndarray
            training data trials x channels x samples
        """
        X_train = self._fit_normalize_pytorch(X_train)
        return X_train

    def normalize(self, x):
        """Applies z-score normalization on input and returns 
        normalized data 

        Parameters
        ----------
        x : numpy ndarray
            input to be normalized by the training data statistics 

        Returns
        -------
        numpy ndarray
            normalized data with same shape as input
        """
        if self.do_normalize:
            return self.model.normalize(x)

    def output_shape(self):
        """Reutns the model's output shape 

        Returns
        -------
        int
            dimension of last model's layer
        """
        return self.model.output_shape    
    
    def get_device(self):
        if self.model.device:
            return self.model.device
        else:
            raise TypeError("The device for evaluation is not set yet")

    def _compile(self,
                 khsara, 
                 optimizer, 
                 metrics, 
                 loss_weights, 
                 schedule, 
                 classes):
        """Uses the model's inner method compile() to compile the model's
        training configuration.

        Parameters
        ----------
        khsara : str | dict
           loss configuration
        optimizer : str | dict
            optimizer configuration
        metrics : list : str
            metrics names
        loss_weights : list : floats
            loss weight when using a combination of losses
        schedule : dict
            learning rate scheduler configuration
        classes : int
            number of classes in dataset
        """
        self.model.compile(loss=khsara,
                            optimizer=optimizer,
                            metrics=metrics,
                            loss_weights=loss_weights,
                            scheduler=schedule,
                            classes=classes
                            )                      

    def _fit_normalize_pytorch(self, X_train):
        """Estimate mean and variance from training data and
        normalize them.
        Uses the model method set_scale()
        Parameters
        ----------
        X_train : numpy ndarray
            training data

        Returns
        -------
        numpy ndarray
            normalized training data
        """
        return self.model.set_scale(X_train)

    def _get_metrics(self, classes):
        """Get a list of default suitable metrics for training according to the numbder
        of classes in dataset.
        
        - Binary classification : Accuracy, AUC, PR AUC, Precision, Recall. ECE, MCE, F1-Score
        - Multi Class : Accuracy only.

        Parameters
        ----------
        classes : int
            number of classes in the dataset
        
        Returns
        -------
        List of metrics
        """        
        if classes == 2:
            metrics = metrics_binary()
        else:
            metrics = ['accuracy'] # how about more metrics than accuracy  for multiclass?

        return metrics

    def _get_loss(self, classes):
        """Get default suitable Loss name according to the number of classes in dataset.
        
        - Binary Classification : binary_crossentropy
        - Multi Class : sparse_categorical_crossentropy | categorical_crossentropy

        Parameters
        ----------
        classes : int
            number of classes in the dataset

        Returns
        -------
        str : name of Loss
        """
        if classes == 2:
            return 'binary_crossentropy'
        elif 'fit' in self.config:
            if 'augment' in self.config['fit']:
                return 'categorical_crossentropy'
        else:
            return 'sparse_categorical_crossentropy'

    def _default_compile(self, classes):
        """Default model compile configuration, used when no compile
        config is passed to evaluation.
        
        - Loss: "binary_crossentropy" or "sparse_categorical_crossentropy"
        - Optimizer : "adam"
        - Metrics : binary metrics or "accuracy"       

        Parameters
        ----------
        classes: int
            number of classes in the dataset

        Returns
        ------
        khsara (loss in Arabic): str
            loss function optimized during training
        optimizer : str ("adam")
            default optimizer congfiguration to be built later
        metrics : list : str 
            metrics measuring the model's performance in evaluaion
        """
        khsara  = self._get_loss(classes)
        metrics = self._get_metrics(classes)
        optimizer = "adam" 
        return khsara, optimizer, metrics

