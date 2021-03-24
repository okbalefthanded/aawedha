from tensorflow.keras.layers.experimental import preprocessing
from aawedha.utils.utils import log, get_gpu_name, init_TPU, time_now
from sklearn.metrics import roc_curve, confusion_matrix
from aawedha.utils.evaluation_utils import class_weights
from aawedha.evaluation.checkpoint import CheckPoint
from aawedha.models.utils_models import freeze_model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from aawedha.io.base import DataSet
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import pickle
import abc
import os


class Evaluation(object):
    """Base class for evaluations

        Evaluation defines and control the main process for training and
        testing a given model on a given dataset following a certain
        configuration.

        Attributes
        ----------
        dataset : DataSet instance
            a dataset from the available sets available to run evaluation on

        model : Keras Model instance
            the model to train/test on the dataset

        partition : list of 2 or 3 integers
            configuration for data partitioning into train/validation/test subset
            default a 3 integers list of:
                In case of a single dataset without an independent Test set
                - (folds/L, folds/M, folds/N) L+M+N = total trials in dataset
                for SingleSubject evaluation
                - (L_subjects, M_subjects, N_subjects) L+M+N = total subjects
                in dataset for CrossSubject evaluation
            a 2 integers list of:
                In case of a dataset with an independent Test set
                - (folds/L, folds/M) L+M = T total trials in dataset for
                SingleSubject evaluation
                - (L_subjects, M_subjects) L+M = S total subjects in dataset
                for CrossSubject evaluation

        folds : a list of 3 1d numpy array
            indices of trials(SingleSubject evaluation)/subjects
            (CrossSubjects evaluation) for each fold

        verbose : int
            level of verbosity for model at fit method ,
            0 : silent, 1 : progress bar, 2 : one line per epoch

        lg : bool
            if True uses logger to log experiment configurations and results,
            default False

        logger : logger
            used to log evaluation results

        predictions : nd array of predictions
            models output for each example on the dataset :
                - SingleSubject evaluation : subjects x folds x Trials x dim
                - CrossSubject evaluation : folds x Trials x dim

        cm : list
            confusion matrix per fold

        results : dict of evaluation results compiled from models performance
        on the dataset
            - 'acc' : 2d array : Accuracy for each subjects on each folds
            (subjects x folds)
            - 'acc_mean' : double : Accuracy mean over all subjects and
            folds
            - 'acc_mean_per_fold' : 1d array : Accuracy mean per fold over
            all subjects
            - 'acc_mean_per_subj' : 1d array : Accuracy mean per Subject over
            all folds [only for SingleSubject evaluation]
            For binary class tasks :
            - 'auc' : 2d array : AUC for each subjects on each folds
            (subjects x folds)
            - 'auc_mean' : double :  AUC mean over all subjects and
            folds
            - 'auc_mean_per_fold' :  1d array : AUC mean per fold over
            all subjects
            - 'auc_mean_per_subj' :  AUC mean per Subject over all folds
                [only for SingleSubject evaluation]
            - 'tpr' : 1d array : True positives rate
            - 'fpr' : 1d array : False positives rate

        n_subjects : int
            number of subjects in dataset if the train and test set have same
            subjects, the sum of both, otherwise.

        model_history : list
            Keras history callbacks

        model_config : dict of model configurations,
        used in compile() and fit().
            compile :
            - loss : str : loss function to optimize during training
                - default  : 'categorical_crossentropy'
            - optimizer : str | Keras optimizer instance : SGD optimizer
                - default : 'adam'
            - metrics : list : str | Keras metrics : training metrics
                - default : multiclass tasks ['accuracy']
                            binary tasks ['accuracy', AUC()]
            fit :
            - batch : int : batch size
                - default : 64
            - epochs : int : training epochs
                - default : 300
            - callbacks : list : Keras model callbacks
                - default : []

        model_compiled : bool, flag for model state
            default : False

    """

    def __init__(self, dataset=None, model=None, partition=None, folds=None,
                 verbose=2, lg=False, debug=False):
        """
        """
        self.dataset = dataset
        self.partition = partition
        self.folds = folds
        self.model = model
        self.predictions = []
        self.cm = []  # confusion matrix per fold
        self.results = {}
        self.model_history = {}
        self.verbose = verbose
        self.n_subjects = self._get_n_subjects()
        self.log = lg
        if self.log:
            if dataset:
                title = dataset.title
            else:
                title = ''
            dataset_folder = f"aawedha/logs/{title}"
            now = datetime.datetime.now().strftime('%c').replace(' ', '_')
            if not os.path.isdir(dataset_folder):
                os.mkdir(dataset_folder)
            # f = 'aawedha/logs/' + '_'.join([self.__class__.__name__,
            #                                title, now, '.log'])
            f = dataset_folder + '_'.join([self.__class__.__name__,
                                           title, now, '.log'])
            self.logger = log(fname=f, logger_name='eval_log')
        else:
            self.logger = None
        self.model_compiled = False
        self.model_config = {}
        self.initial_weights = []
        #  self.normalizer = None # preprocessing.Normalization(axis=(1, 2))
        self.current = None
        self.debug = debug
        self.log_dir = None


    def __str__(self):
        name = self.__class__.__name__
        model = self.model.name if self.model else 'NotSet'
        info = (f'Type: {name}',
                f'DataSet: {str(self.dataset)}',
                f'Model: {model}')
        return '\n'.join(info)

    @abc.abstractmethod
    def generate_split(self, nfolds):
        """Generate train/validation/test split
            Overridden in each type of evaluation : SingleSubject | CrossSubject | CrossSet
        """
        pass

    @abc.abstractmethod
    def run_evaluation(self):
        """Main evaluation process
            Overridden in each type of evaluation : SingleSubject | CrossSubject | CrossSet 
        """
        pass

    @abc.abstractmethod
    def get_operations(self):
        pass

    @abc.abstractmethod
    def get_folds(self):
        pass

    def resume(self, run=False, savecsv=False):
        """Resume evaluation from where it was interrupted

        Parameters
        ----------
        run : bool
            if true resume evaluation at checkpoint

        Returns
        -------
        instance of evaluation
        """
        chk = self.reset(chkpoint=True)
        device = self._get_device()
        if device == 'TPU':
            self._compile_model()
        if run:
            self.run_evaluation(pointer=chk, check=True, savecsv=savecsv)
        return self

    def set_dataset(self, dt=None):
        """Instantiate dataset with dt

        Parameters
        ----------
        dt : dataset instance
            a dataset object

        Returns
        -------
        no value
        """
        self.dataset = dt

    def measure_performance(self, Y_test, probs, perf):
        """Measure model performance on a dataset

        Calculates model performance on metrics and sets Confusion Matrix for
        each fold

        Parameters
        ----------
        Y_test : 2d array (n_examples x n_classes)
            true class labels in Tensorflow format

        probs : 2d array (n_examples x n_classes)
            model output predictions as probability of belonging to a class

        Returns
        -------
            dict of performance metrics : {metric : value}
        """
        self.predictions.append(probs)  # ()
        results = dict()        
        # classes = Y_test.max()
        if Y_test.ndim == 2:
            Y_test = Y_test.argmax(axis=1)
            probs = probs[:, 1]

        classes = np.unique(Y_test).size       

        # self.cm.append(confusion_matrix(Y_test, preds))

        for metric, value in zip(self.model.metrics_names, perf):
            results[metric] = value

        if classes == 2:
            fp_rate, tp_rate, thresholds = roc_curve(Y_test, probs)
            viz = {'fp_threshold': fp_rate, 'tp_threshold': tp_rate}
            results['viz'] = viz
            preds = np.zeros(len(probs))
            preds[probs.squeeze() > .5] = 1.
        else:
            preds = probs.argmax(axis=-1)     
        
        self.cm.append(confusion_matrix(Y_test, preds))

        return results

    def results_reports(self, res, tfpr={}):
        """Collects evaluation results on a single dict

        Parameters
        ----------
        res : dict
            contains models performance
            - 'acc' : list
                - SingleSubject : Accuracy of each subject on all folds
                - CrossSubject :  Accuracy on all folds
            - 'auc' : list (only for binary class tasks)
                - SingleSubject : AUC of each subject on all folds
                - CrossSubject :  AUC on all folds

        tfpr : dict (only for binary class tasks)
            - 'fp' : False positives rate on all fold
            - 'tp' : True positives rate on all fold

        Returns
        -------
        results : dict of evaluation results compiled from models performance
        on the dataset
            - 'acc' : 2d array : Accuracy for each subjects on each folds
            (subjects x folds)
            - 'acc_mean' : double : Accuracy mean over all subjects and
            folds
            - 'acc_mean_per_fold' : 1d array : Accuracy mean per fold over
            all subjects
            - 'acc_mean_per_subj' : 1d array : Accuracy mean per Subject over
            all folds [only for SingleSubject evaluation]
            For binary class tasks :
            - 'auc' : 2d array : AUC for each subjects on each folds
            (subjects x folds)
            - 'auc_mean' : double :  AUC mean over all subjects and
            folds
            - 'auc_mean_per_fold' :  1d array : AUC mean per fold over
            all subjects
            - 'auc_mean_per_subj' :  AUC mean per Subject over all folds
            [only for SingleSubject evaluation]
            - 'tpr' : 1d array : True posititves rate
            - 'fpr' : 1d array : False posititves rate
        """

        if isinstance(self.dataset.epochs, np.ndarray):
            folds = len(self.folds)
            # subjects = self._get_n_subjects()
            # subjects = len(self.predictions)
            examples = len(self.predictions[0])
            dim = len(self.predictions[0][0])

            if self.__class__.__name__ == 'CrossSubject' or self.__class__.__name__ == 'CrossSet':
                self.predictions = np.array(self.predictions).reshape(
                    (folds, examples, dim))
            elif self.__class__.__name__ == 'SingleSubject':
                self.predictions = np.array(self.predictions).reshape(
                    (self.n_subjects, folds, examples, dim))

        res = self._aggregate_results(res)
        """
        metrics = list(res.keys())
        if self.dataset.get_n_classes() == 2:
            metrics.remove('viz')

        for metric in metrics:
            res[metric + '_mean'] = np.array(res[metric]).mean()
            res[metric + '_mean_per_fold'] = np.array(res[metric]).mean(axis=0)
            if np.array(res[metric]).ndim == 2:
                res[metric +
                    '_mean_per_subj'] = np.array(res[metric]).mean(axis=1)
        """
        res = self._update_results(res)
        return res

    def save_model(self, folderpath=None, save_frozen=False):
        """Save trained model in HDF5 format
        Uses the built-in save method in Keras Model object.
        model name will be: folderpath/modelname_paradigm_dataset.h5

        Parameters
        ----------
        folderpath : str
            folder location where the model will be saved
            default : 'aawedha/trained

        Returns
        -------
        no value
        """
        if not os.path.isdir('trained'):
            os.mkdir('trained')
        if not folderpath:
            folderpath = 'trained'
        prdg = self.dataset.paradigm.title
        dt = self.dataset.title
        # filepath = folderpath + '/' + '_'.join([self.model.name, prdg, dt, '.h5'])
        filepath = os.join.path(folderpath, '_'.join([self.model.name, prdg, dt, '.h5']))
        self.model.save(filepath)

        if save_frozen:
            frozen_path = freeze_model(self.model, folderpath)

    def set_model(self, model=None, model_config={}):
        """Assign Model and model_config

        Parameters
        ----------
        model : Keras model
            model to be trained and evaluated, selected from the available ones.
            default : None

        model_config : dict of model configurations, used in compile() and fit().
            compile :
            - loss : str : loss function to optimize during training
                - default  : 'categorical_crossentropy'
            - optimizer : str | Keras optimizer instance : SGD optimizer
                - default : 'adam'
            - metrics : list : str | Keras metrics : training metrics
                - default : multiclass tasks ['accuracy']
                            binary tasks ['accuracy', AUC()]
            fit :
            - batch : int : batch size
                - default : 64
            - epochs : int : training epochs
                - default : 300
            - callbacks : list : Keras model callbacks
                - default : []
            default : empty dict, attributed will be set at compile/fit calls

        Returns
        -------
        no value
        """
        if model_config:
            self.set_config(model_config)
            # self.model_config = model_config

        # UNRESOLVED : due to TF disconnected graph issue that will occur
        # if we use the below code, we'll revert back to the old code where
        # we didn't modify any layer of the model, i.e. it's set as it is
        # defined at/outside this method call.
        '''
        if type(model.layers[0].input_shape) is list:
            # model created using Functional API
            input_shape = model.layers[0].input_shape[0][1:]
        else:
            # model created using Sequential class
            input_shape = model.layers[0].input_shape[1:]

        if type(model.layers[0]).__name__ is 'InputLayer':
            layer_input = 1
        else:
            layer_input = 0

        input_type = type(model.layers[layer_input]).__name__
        
        if input_type is not 'Normalization':
            model_name = f'{model.name}_norm_'
            model_input = tf.keras.Input(shape=input_shape[1:])
            norm = self.normalizer(model_input)
            hidden = tf.keras.layers.Reshape(input_shape)(norm)
            for layer in model.layers:
                if type(layer).__name__ is 'InputLayer':
                    continue
                hidden = layer(hidden)
            self.model = tf.keras.models.Model(inputs=model_input, outputs=hidden, name=model_name)
        else:
            self.model = model
        '''
        self.model = model
        '''
        for layer in self.model.layers:
            if type(layer).__name__ is 'Normalization':
                self.normalizer = layer
        '''
        self.initial_weights = self.model.get_weights()

    def set_config(self, model_config):
        """Setter for model_config

        Parameters
        ----------
        model_config : dict
            dict of model configurations, used in compile() and fit()
            compile :
            - loss : str : loss function to optimize during training
                - default  : 'categorical_crossentropy'
            - optimizer : str | Keras optimizer instance : SGD optimizer
                - default : 'adam'
            - metrics : list : str | Keras metrics : training metrics
                - default : multiclass tasks ['accuracy']
                            binary tasks ['accuracy', AUC()]
            fit :
            - batch : int : batch size
                - default : 64
            - epochs : int : training epochs
                - default : 300
            - callbacks : list : Keras model callbacks
                - default : []
            default : empty dict, attributed will be set at compile/fit calls

        Returns
        -------
        no value
        """
        self.model_config = model_config

    def log_experiment(self):
        """Write in logger, evaluation information after completing a fold
        Message format :
        date-time-logger_name-logging_level-{fold|subject}-{ACC|AUC}-performance

        Parameters
        ----------
        None

        Returns
        -------
        no value
        """
        s = ['train', 'val', 'test']
        '''
        if self.dataset:
            data = f' Dataset: {self.dataset.title}'
            if isinstance(self.dataset.epochs, list):
                duration = f' epoch duration:{self.dataset.epochs[0].shape[0] / self.dataset.fs} sec'
                data_shape = f'{len(self.dataset.epochs), self.dataset.epochs[0].shape} list'
            else:
                duration = f' epoch duration:{self.dataset.epochs.shape[1] / self.dataset.fs} sec'
                data_shape = f'{self.dataset.epochs.shape}'
        else:
            data = ''
            duration = '0'
            data_shape = '[]'
        '''
        data, duration, data_shape = self._dataset_info()

        prt = 'Subjects partition ' + \
              ', '.join(f'{s[i], self.partition[i]}' for i in range(
                  len(self.partition)))
        model = f'Model: {self.model.name}'
        model_config = f'Model config: {self._get_model_configs_info()}'
        device = self._get_device()
        if device == 'GPU':
            compute_engine = get_gpu_name()
        else:
            compute_engine = 'TPU'
        
        mode = ''
        if hasattr(self, 'mode'):
            mode = f"Cross Set Mode: {self.mode}"

        exp_info = ' '.join([data, duration, data_shape, prt, model,
                             model_config, compute_engine, mode])
        self.logger.debug(exp_info)

    def reset(self, chkpoint=False):
        """Reset Attributes and results for a future evaluation with
            different model and same partition and folds

        if chkpoint is passed, Evaluation attributes will be set to
        the state at where the evaluation was interrupted, this is used
        for operations resume

        Parameters
        ----------
        chkpoint : bool
            load checkpoint to resume where the evaluation has stopped earlier,
            reset the evaluation for future use with different configs
            otherwise

        Returns
        -------
        chk : CheckPoint instance
            checkpoint object to set Evaluation state back where it was
            interrupted
        """
        chk = None
        if chkpoint:
            chk = self._load_checkpoint()

        if chk:
            self.model = load_model(chk.model_name)
            self.folds = chk.folds
            self.partition = chk.partition
            self.predictions = chk.predictions
            self.cm = chk.cm
            self.results = chk.results
            self.model_history = chk.model_history
            self.model_compiled = True
            self.model_config = chk.model_config
            self.current = chk.current
            self.log = chk.log
            self.logger = log(fname=chk.logger, logger_name='eval_log')
            self.verbose = chk.verbose
            self.initial_weights = chk.initial_weights
            if hasattr(chk, 'mode'):
                self.mode = chk.mode
            # self.normalizer = chk.normalizer

        else:
            self.model = None
            self.predictions = []
            self.cm = []  # confusion matrix per fold
            self.results = {}
            self.model_history = {}
            self.model_compiled = False
            self.model_config = {}
            self.initial_weights = []

        return chk

    def reset_weights(self):
        """reset model's weights to initial state (model's creation state)
        """
        self.model.set_weights(self.initial_weights)
        # layer 0 : Normalization
        # layer 1 : Reshape
        # layer 2 : Model
        # self.model.layers[2].set_weights(self.initial_weights)

    def _equal_subjects(self):
        """Test whether dataset's train_epochs and test_epochs has same number
        of subjects

        Parameters
        ----------
        None

        Returns
        -------
        bool : True if number of subjects in training data equals the number of
        subjects in test data False otherwise
        """
        test_epochs = 0
        train_epochs = len(self.dataset.epochs)
        if hasattr(self.dataset, 'test_epochs'):
            test_epochs = len(self.dataset.test_epochs)
        return train_epochs == test_epochs

    def _get_n_subjects(self):
        """Return number of subjects in dataset

        Parameters
        ----------
        None

        Returns
        -------
        int : number of subjects if train/test subjects is the same
                their sum otherwise
        """
        if self.dataset:
            test_epochs = len(self.dataset.test_epochs) if hasattr(
                self.dataset, 'test_epochs') else 0
            if self._equal_subjects():
                return len(self.dataset.epochs)
            else:
                return len(self.dataset.epochs) + test_epochs
        else:
            return 0

    def _compile_model(self):
        """Compile model using specified model_config, default values otherwise

        Sets model_compiled attribute to true after successful model
            compilation

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        device = self._get_device()
        khsara, optimizer, metrics = self._get_compile_configs()

        if device != 'TPU':
            self.model.compile(loss=khsara,
                               optimizer=optimizer,
                               metrics=metrics
                               )
        else:
            strategy = init_TPU()
            with strategy.scope():
                self.model = tf.keras.models.clone_model(self.model)
                metrics = self._get_metrics()
                self.model.compile(loss=khsara,
                                   optimizer=optimizer,
                                   metrics=metrics
                                   )
        self.model_compiled = True

    def _eval_model(self, X_train, Y_train, X_val,
                    Y_val, X_test, Y_test, cws):
        """Train model on train/validation data and predict its output on test data

        Run model's fit() and predict() methods

        Parameters
        ----------
        X_train : nd array
            training data
        Y_train : nd array
            true training data labels
        X_val : nd array
            validation data
        X_test : nd array
            test data

        cws : dict (int:float)
            class weights

        Returns
        -------
        history : Keras history callback
            the model's loss and metrics performances per epoch

        probs : 2d array (n_examples x n_classes)
            model's output on test data as probabilities of belonging to
            each class

        perf : array
            model's performance on test data: accuracy and loss
        """
        batch, ep, clbs = self._get_fit_configs()

        device = self._get_device()
        '''
        if device != 'CPU':
            X_train, X_test, X_val = self._transpose_split(
                [X_train, X_test, X_val])
        '''
        if X_val is None:
            val = None
        else:
            val = (X_val, Y_val)
        #
        self.reset_weights()
        self._normalize(X_train)

        history = {}
        spe = None
        if device == 'TPU':
            spe = X_train.shape[0] // batch
            '''
            if format == 'channels_first':
                spe = X_train.shape[0] // batch
            else:
                spe = X_train.shape[-1] // batch
            '''
        probs, perf = None, None

        history = self.model.fit(X_train, Y_train,
                                 batch_size=batch,
                                 epochs=ep,
                                 steps_per_epoch=spe,
                                 verbose=self.verbose,
                                 validation_data=val,
                                 class_weight=cws,
                                 callbacks=clbs)
        
        if isinstance(X_test, np.ndarray):
            probs = self.model.predict(X_test)
            perf = self.model.evaluate(X_test, Y_test, verbose=0)
        
        return history, probs, perf

    def _eval_split(self, split={}):
        """Evaluate the performance of the model on a give split

        Parameters
        ----------
        split : dict
            ndarrays to evaluate: train/val/test data and labels

        Returns
        -------
        rets: dict of performance metrics
        """
        rets = []
        X_train, Y_train = split['X_train'], split['Y_train']
        X_test, Y_test = split['X_test'], split['Y_test']
        X_val, Y_val = split['X_val'], split['Y_val']
        #
        cws = class_weights(Y_train)
        # evaluate model on subj on all folds
        self.model_history, probs, perf = self._eval_model(X_train, Y_train,
                                                           X_val, Y_val,
                                                           X_test, Y_test,
                                                           cws)
        if isinstance(X_test, np.ndarray):
            rets = self.measure_performance(Y_test, probs, perf)
        return rets

    def _get_compile_configs(self):
        """Returns default model compile configurations as tuple

        Parameters
        ----------
        None

        Returns
        -------
        khsara (loss in Arabic): str
            loss function optimized during training
        opt : str
            optimizer
        mets : list : str | keras metrics
            metrics
        """   
        if 'compile' in self.model_config:
            metrics = self.model_config['compile']['metrics']
            khsara = self.model_config['compile']['loss']
            optimizer = self.model_config['compile']['optimizer']
        else:
            khsara = self._get_loss()
            if self._get_device() != 'TPU':
                metrics = self._get_metrics()
            else:
                metrics = []            
            optimizer = 'adam'
            # set config for checkpoint use            

        return khsara, optimizer, metrics

    def _normalize(self, X_train):
        """Adapt normalization layer if it's inside the model
        Parameters
        ----------
        X_train : ndarray
            training data n_samples x channels x samples
        """
        for layer in self.model.layers:
            if type(layer).__name__ is "Normalization":
                layer.adapt(X_train)

    def _get_fit_configs(self):
        """Returns fit configurations as tuple

        Parameters
        ----------
        None

        Returns
        -------
        batch : int
            batch size
        ep : int
            epochs number
        clbks : list
            keras callbacks added to be watched during training
        """
        if 'fit' in self.model_config:
            batch = self.model_config['fit']['batch']
            ep = self.model_config['fit']['epochs']
            clbks = self.model_config['fit']['callbacks']
            # format = self.model_config['fit']['format']
        else:
            batch = 64
            ep = 300
            clbks = []
            # format = 'channels_first'
        
        # K.set_image_data_format(format)
        
        if self.debug:
            # logdir = os.path.join("aawedha/debug", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            debug_dir = os.path.join("aawedha", "debug")
            self.log_dir = os.path.join(debug_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            if not os.path.isdir(self.log_dir):
                os.mkdir(self.log_dir)
            clbks.append(tf.keras.callbacks.TensorBoard(self.log_dir))
        
        return batch, ep, clbks #, format

    def _get_model_configs_info(self):
        """Construct a logging message to be added to logger at evaluation beginning

        the message details the models configuration used for model

        Parameters
        ----------
        None

        Returns
        -------
        model_config : str
            model's configuration
        """
        khsara, opt, mets = self._get_compile_configs()
        batch, ep, clbs = self._get_fit_configs()
        model_config = f' Loss: {khsara} | Optimizer: {opt} | \
                            metrics: {mets} | batch_size: {batch} | \
                            epochs: {ep} | \
                            callbacks: {clbs}'
        return model_config

    def _get_metrics(self):
        """Get a list of default suitable metrics for training according to the numbder
        of classes in dataset.
        
        - Binary classification : Accuracy, AUC, True Positives, False Positives, True Negatives,
            False Negatives, Precision, Recall.
        - Multi Class : Accuracy only.
        
        Returns
        -------
        List of metrics
        """
        classes = self._get_classes()
        
        if classes == 2:
            metrics = ['accuracy',
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.TruePositives(name='tp'),
                        tf.keras.metrics.FalsePositives(name='fp'),
                        tf.keras.metrics.TrueNegatives(name='tn'),
                        tf.keras.metrics.FalseNegatives(name='fn'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')
                        ]
        else:
            metrics = ['accuracy']

        return metrics

    def _get_loss(self):
        """Get default suitable Loss name according to the number of classes in dataset.
        
        - Binary Classification : binary_crossentropy
        - Multi Class : sparse_categorical_crossentropy

        Returns
        -------
        str : name of Loss
        """
        classes = self._get_classes()
        if classes == 2:
            return 'binary_crossentropy'
        else:
            return 'sparse_categorical_crossentropy' 

    def _get_classes(self):
        """Number of classes in DataSet
        Returns
        -------
        int : number of classes if dataset is set, 0 otherwise.
        """
        if self.dataset:
            return self.dataset.get_n_classes()
        else:
            return 0        

    def _assert_partition(self, excl=False):
        """Assert if partition to be used do not surpass number of subjects available
        in dataset

        Parameters
        ----------
        excl : bool
            flag indicating whether the target subject is excluded from
            evaluation

        Returns
        -------
        bool : True if number of subjects is less than the sum of parition
            False otherwise
        """
        subjects = self._get_n_subjects()
        prt = np.sum(self.partition)
        return subjects < prt

    @staticmethod
    def _load_checkpoint():
        """load saved checkpoint to resume evaluation
        Parameters
        ----------
        no parameters

        Returns
        -------
        checkpoint :
            a checkpoint of a saved evaluation
        """
        file_name = 'aawedha/checkpoints/current_CheckPoint.pkl'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                chkpoint = pickle.load(f)
        else:
            raise FileNotFoundError
        return chkpoint

    def _has_val(self):
        """test if evaluation has partition for validation phase

        Returns
        -------
        int :
           if 0 no validation, otherwise number of parts from dataset to use
            in validation
        """
        train = self.partition[0]
        if len(self.partition) == 3:
            test = self.partition[2]
        else:
            test = 1

        return self._get_n_subjects() - train - test

    @staticmethod
    def _transpose_split(arrays):
        """Transpose input Data to be prepared for NCHW format
        N : batch (assigned at fit), C: channels here refers to trials,
        H : height here refers to EEG channels, W : width here refers to samples

        Parameters
        ----------
        arrays: list of data arrays
            - Training data in 1st position
            - Test data in 2nd position
            - if not none, Validation data
        Returns
        ------- 
        list of arrays same order as input
        """
        for i, arr in enumerate(arrays):
            if isinstance(arr, np.ndarray):
                arrays[i] = arr.transpose((2, 1, 0))
                # trials , channels, samples
        return arrays

    @staticmethod
    def _aggregate_results(res):
        """Aggregate subject's results from folds into a single list

        Parameters
        ----------
        results : list of dict
            each element in the list is a dict of performance
            values in a fold

        Returns
        -------
        dict of performance metrics
        """
        results = dict()
        if type(res) is list:
            metrics = res[0].keys()
        else:
            metrics = res.keys()
        for metric in metrics:
            tmp = []
            for fold in res:
                tmp.append(fold[metric])
            results[metric] = tmp

        return results

    def _update_results(self, res):
        """Add metrics results mean to results dict.

        Parameters
        ----------
        res : dict
            dictionary of metrics results
        Returns
        -------
        res : dict
            updated dictionary with metrics mean fields.
        """
        metrics = list(res.keys())
        if isinstance(self.dataset, DataSet):
            classes = self.dataset.get_n_classes()
        else:
            classes = self.target.get_n_classes()

        if classes == 2:
            metrics.remove('viz')

        for metric in metrics:
            res[metric + '_mean'] = np.array(res[metric]).mean()
            res[metric + '_mean_per_fold'] = np.array(res[metric]).mean(axis=0)
            if np.array(res[metric]).ndim == 2:
                res[metric +
                    '_mean_per_subj'] = np.array(res[metric]).mean(axis=1)
        
        return res

    def _get_device(self):
        """Returns compute engine : GPU / TPU

        Returns
        -------
        str
            computer engine for training
        """
        # test if env got GPU
        device = 'GPU'  # default
        if 'device' in self.model_config:
            return self.model_config['device']
        else:
            devices = [
                dev.device_type for dev in tf.config.get_visible_devices()]
            if 'GPU' not in devices:
                device = 'CPU'
            return device

    def _log_operation_results(self, op_ind, op_results):
        """Save an evaluation iteration results in logger.

        Parameters
        ----------
        op_ind : int
            operation index : 
             - subject number in case of SingleSubject evaluation.
             - fold index in case of both CrossSubject and CrossSet evaluation.
        op_results : dict
            operation metric results: accuracy and auc (if binary classification)
        """         
        msg = f" Subj : {op_ind+1} ACC: {np.array(op_results['accuracy'])*100}"
        if 'auc' in op_results:
            msg += f" AUC: {np.array(op_results['auc'])*100}"
        msg += f' Training stopped at epoch: {self.model_history.epoch[-1]}'
        self.logger.debug(msg)        
    
    def _log_results(self):
        """Log metrics means after the end of an evaluation to logger"""
        means = [f'{metric}: {v}' for metric,
                 v in self.results.items() if 'mean' in metric]
        self.logger.debug(' / '.join(means))

    
    def _dataset_info(self):
        """Collect informations on evaluation dataset, which will be used in logging.
        Returns
        -------
        data : str
            dataset title
        duration : str
            epoch length in seconds
        data_shape : str
            dimension of data : subjects x samples x channels x trials
        """
        if self.dataset:
            data = f' Dataset: {self.dataset.title}'
            if isinstance(self.dataset.epochs, list):
                duration = f' epoch duration:{self.dataset.epochs[0].shape[0] / self.dataset.fs} sec'
                data_shape = f'{len(self.dataset.epochs), self.dataset.epochs[0].shape} list'
            else:
                duration = f' epoch duration:{self.dataset.epochs.shape[1] / self.dataset.fs} sec'
                data_shape = f'{self.dataset.epochs.shape}'
        else:
            data = ''
            duration = '0'
            data_shape = '[]'

        return data, duration, data_shape
        
    
    def _savecsv(self, folder=None):
        """Save evaluation results in a CSV file as Pandas DataFrame

        Parameters
        ----------
        folder : str
            results files will be stored inside folder, if None, a default folder inside aawedha is used.

        Returns
        -------
        None
        """
        if not folder:
            folder = 'aawedha/results'
            if not os.path.isdir(folder):
                os.mkdir(folder)

        metrics = {'accuracy', 'auc'}
        results_keys = set(self.results)
        metrics = metrics.intersection(results_keys)
        subjects = range(self._get_n_subjects())
        rows = [f'S{s+1}' for s in subjects]
        rows.append('Avg')
        if isinstance(self.dataset, DataSet):
            dataset = self.dataset.title
        else:
            dataset = self.target.title
        
        evl = self.__class__.__name__
        columns = []
        
        if evl == 'CrossSubject' or 'CrossSet':
            columns = ['Fold 1']
        elif evl == 'SingleSubject':
            nfolds = len(self.results['accuracy'][0])
            # columns = [f'Fold {fld+1}' for fld, _ in enumerate(evl.folds)]
            columns = [f'Fold {fld+1}' for fld in range(nfolds)]

        columns.extend(['Avg', 'Std', 'Sem'])
        date = time_now()
        for metric in metrics:
            acc = np.array(self.results[metric]).round(3) * 100
            if acc.ndim == 1:
                acc_mean = acc
                std = np.array(self.results[metric]).std().round(3) * 100
                std = np.tile(std, len(rows) - 1)
            else:
                acc_mean = np.array(self.results[metric]).mean(
                    axis=1).round(3) * 100
                std = np.array(self.results[metric]).std(axis=1).round(3) * 100
            sem = np.round(std / np.sqrt(len(self.results[metric])), 3)
            values = np.column_stack((acc, acc_mean, std, sem))
            values = np.vstack((values, values.mean(axis=0).round(3)))
            df = pd.DataFrame(data=values, index=rows, columns=columns)
            df.index.name = f"{self.model.name} / {metric}"
            fname = f"{folder}/{evl}_{dataset}_{metric}_{date}.csv"
            df.to_csv(fname, encoding='utf-8')
