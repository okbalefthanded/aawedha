from aawedha.optimizers.utils_optimizers import optimizer_lib, get_optimizer
from aawedha.evaluation.evaluation_utils import metrics_by_lib
# from aawedha.models.pytorch.torchmodel import TorchModel
from aawedha.models.builder_model import build_learner
from aawedha.models.utils_models import model_lib
from aawedha.utils.utils import init_TPU
import tensorflow as tf


class Learner:
    """Wrapper class to unify training and inference operations for different
    Frameworks.
    """
    def __init__(self, model=None, compiled=False, weights={},
                config={}, history=[], normalize=True,
                name=None, model_type=None):
        """Constructor
        initiliaze Learner object with empty attributes.
        The object update will be made at compile method call.

        Parameters
        ----------
        model : Keras Model | TorchModel, optional
            the model instance that will be trained, by default None
        compiled : bool, optional
            True if the model is compiled, by default False
        weights : {list, dict}, optional
            the model's weights at creation, used to keep the same start 
            point in later evaluations.
            - List: Keras model initial weights.
            - dict : TorchModel module initial state.
            by default {}
        config : dict, optional
            compile and training configuration, by default {}
        history : a list of : {Keras History object, dict} optional
            Record of training/validation metrics and loss values per epoch, by default []
            - History object: if the model is a Keras Model object.
            - dict: if the model is a TorchModel object.
        normalize : bool, optional
            if True, normalize inputs before training and evaluation, by default True
        name : str, optional
            model name, by default None
        model_type :  {'keras', 'pytorch'}, optional
            model framework, by default None
        """
        # self._init_model(model, model_type)
        self.model    = model        
        self.compiled = compiled        
        self.config   = config
        self.history  = history
        self.name = name
        self.type = model_type
        self.initial_weights = weights
        self.do_normalize    = normalize        

    def set_model(self, model, engine):
        """model setter
        - Keras   : assign the model to Learner model attribute
        - Pytorch : create a TorchModel (or subclass) object, then assign model to
        the TorchModel module attribute.

        Parameters
        ----------
        model : {Keras Model, Pytorch nn Module}
            model object for training and evalution.
        engine : {'keras', 'pytorch'}
            model framework.
        """
        if engine == "pytorch":
            if not self.model:
                self.model = build_learner(self.config['compile'])
                self.model.module = model
        elif engine == "keras":
            self.model = model
        self.name = model.name

    def compile(self, device, classes):
        """Compile model: create training configuration objects
        from description

        Parameters
        ----------
        device : str
            compute hardware: {CPU | GPU | TPU}
        classes : int
            number of classes in the dataset to be trained on
        """
        devices = {'CPU': 'cpu', 'GPU': 'cuda'}
        khsara, optimizer, metrics, schedule = self.get_compile_configs(device, classes)
        if device != 'TPU':
            if self.type == 'keras':
                self._compile_keras(khsara, optimizer, metrics)
            else:
                self.model.set_device(devices[device.upper()])
                self._compile_pytorch(khsara, optimizer, metrics, schedule, classes)
        else:
            self._compile_for_tpu(khsara, optimizer, classes)
        self.compiled = True

    def fit(self, x, y, batch_size, epochs, steps_per_epoch, 
            verbose, validation_data, class_weight, 
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
        steps_per_epoch : {int, None}
            used mainly for training Keras Models in TPUs.
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
        history : {Keras History object, dict} optional
            Record of training/validation metrics and loss values per epoch, by default []
            - History object: if the model is a Keras Model object.
            - dict: if the model is a TorchModel object.
        """
        return self.model.fit(x=x, y=y, 
                              batch_size=batch_size,
                              epochs=epochs, 
                              steps_per_epoch=steps_per_epoch,
                              verbose=verbose, 
                              validation_data=validation_data,
                              class_weight=class_weight, 
                              callbacks=callbacks)

    def predict(self, X):
        preds = []
        if self.type == "keras":
            preds = self.model.predict(X)
        else:
            preds = self.model.predict(X, normalize=self.do_normalize)
        return preds

    def evaluate(self, X, Y, batch_size=32, return_dict=True, verbose=0):
        perfs = {}
        if self.type == "keras":
            perfs =  self.model.evaluate(X, Y, batch_size=batch_size, return_dict=return_dict, 
                                        verbose=0)
        else:
            perfs =  self.model.evaluate(X, Y, batch_size=batch_size, normalize=self.do_normalize,
                                         return_dict=return_dict, verbose=verbose)
        return perfs
    
    def save(self, filepath):
        self.model.save(filepath)

    def reset_weights(self):
        if self.type == "keras":
            self._reset_state_keras()
        else:
            self._reset_state_pytorch()

    def get_compile_configs(self, device, classes):
        """Returns default model compile configurations as tuple

        Returns
        -------
        khsara (loss in Arabic): str
            loss function optimized during training
        opt : str
            optimizer
        mets : list : str | keras metrics
            metrics
        """ 
        schedule = None
        if 'compile' in self.config:
            if 'metrics' in self.config['compile']:
                metrics = self.config['compile']['metrics']
            else:
                metrics = self._get_metrics(classes)
            if 'loss' in self.config['compile']:
                khsara = self.config['compile']['loss']
            else:
                khsara = self._get_loss(classes)
            if 'optimizer' in self.config['compile']:
                optimizer = self.config['compile']['optimizer']
                if isinstance(optimizer, str) or isinstance(optimizer, list):
                    if self.type == "keras":
                        opt_lib = optimizer_lib(optimizer)
                        if opt_lib != 'builtin':
                            optimizer = get_optimizer(optimizer, opt_lib)
            else:
                optimizer = 'adam'
            if 'scheduler' in self.config['compile']:
                schedule = self.config['compile']['scheduler']
        else:
            khsara, optimizer, metrics = self._default_compile(device, classes)           

        return khsara, optimizer, metrics, schedule

    def fit_normalize(self, X_train):
        """Adapt normalization layer if it's inside the model
        Parameters
        ----------
        X_train : ndarray
            training data n_samples x channels x samples
        """
        if self.type == "keras":
            self._fit_normalize_keras(X_train)            
        else:
            X_train = self._fit_normalize_pytorch(X_train)
        return X_train

    def normalize(self, x):
        if self.do_normalize and self.type == "pytorch":
            return self.model.normalize(x)

    def set_type(self):
        self.type = model_lib(type(self.model))

    def output_shape(self):
        
        if self.type == 'keras':
            raise NotImplementedError
        else:
            return self.model.output_shape

    # def _init_model(self, model, model_type):
    #     if model_type == "pytorch":
    #         self.model = TorchModel(module=model)
    
    def _compile_keras(self, khsara, optimizer, metrics):
        self.model.compile(loss=khsara,
                               optimizer=optimizer,
                               metrics=metrics
                               )

    def _compile_pytorch(self, khsara, optimizer, metrics, schedule, classes):
        self.model.compile(loss=khsara,
                               optimizer=optimizer,
                               metrics=metrics,
                               scheduler=schedule,
                               classes=classes
                               )
    
    def _compile_for_tpu(self, khsara, optimizer, classes):
        strategy = init_TPU()
        with strategy.scope():
            self.model = tf.keras.models.clone_model(self.model)
            metrics = self._get_metrics(classes)
            self.model.compile(loss=khsara,
                                optimizer=optimizer,
                                metrics=metrics
                                   )                  

    def _fit_normalize_keras(self, X_train):
        for layer in self.model.layers:
                if type(layer).__name__ == "Normalization":
                    layer.adapt(X_train)        

    def _fit_normalize_pytorch(self, X_train):
        return self.model.set_scale(X_train)

    def _get_metrics(self, classes):
        """Get a list of default suitable metrics for training according to the numbder
        of classes in dataset.
        
        - Binary classification : Accuracy, AUC, True Positives, False Positives, True Negatives,
            False Negatives, Precision, Recall.
        - Multi Class : Accuracy only.
        
        Returns
        -------
        List of metrics
        """        
        if classes == 2:
            metrics = metrics_by_lib(self.type)
        else:
            metrics = ['accuracy']

        return metrics

    def _get_loss(self, classes):
        """Get default suitable Loss name according to the number of classes in dataset.
        
        - Binary Classification : binary_crossentropy
        - Multi Class : sparse_categorical_crossentropy | categorical_crossentropy

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

    def _default_compile(self, device, classes):
        """Default model compile configuration, used when no compile
        config is passed to evaluation.
        """
        khsara = self._get_loss(classes)
        if device != 'TPU':
            metrics = self._get_metrics(classes)
        else:
            metrics = []
        optimizer = "adam" if self.type == "Keras" else "Adam"
        return khsara, optimizer, metrics

    def _reset_state_keras(self):
        if not self.initial_weights: 
            self.initial_weights['model_weights'] = self.model.get_weights()
        else:
            self.model.set_weights(self.initial_weights['model_weights'])
            if len(self.model.optimizer._weights) == 1:
                self.model.optimizer._weights = []
            else:
                self.model.optimizer._weights[-1] = []

    def _reset_state_pytorch(self):
        # if not self.initial_weights: 
        #     self.initial_weights['model_weights'] = self.model.get_weights()
        # else:
        #     self.model.set_weights(self.initial_weights['model_weights'])
        self.model.set_weights()