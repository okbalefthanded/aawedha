from aawedha.evaluation.evaluation_utils import class_weights, labels_to_categorical, metrics_by_lib
from aawedha.utils.utils import log, get_gpu_name, init_TPU, time_now, make_folders
from aawedha.optimizers.utils_optimizers import optimizer_lib, get_optimizer
from aawedha.evaluation.checkpoint import CheckPoint
from aawedha.evaluation.evaluation_utils import aggregate_results
from aawedha.evaluation.mixup import build_mixup_dataset
from sklearn.metrics import roc_curve, confusion_matrix
from tensorflow.keras.models import load_model
from aawedha.io.base import DataSet
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
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
                 verbose=2, lg=False, engine="keras", normalize=True, debug=False):
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
            if not os.path.isdir("aawedha/logs"):
                make_folders()
            dataset_folder = f"aawedha/logs/{title}/"
            now = datetime.datetime.now().strftime('%c').replace(' ', '_')
            if not os.path.isdir(dataset_folder):
                os.mkdir(dataset_folder)
            f = dataset_folder + '_'.join([self.__class__.__name__,
                                           title, now, '.log'])
            self.logger = log(fname=f, logger_name='eval_log')
        else:
            self.logger = None
        self.model_compiled = False
        self.model_config = {}
        self.initial_weights = []
        self.normalize = normalize
        self.engine = engine
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
            # probs = probs[:, 1]

        classes = np.unique(Y_test).size

        # self.cm.append(confusion_matrix(Y_test, preds))
        
        for metric, value in zip(self.model.metrics_names, perf):
            results[metric] = value

        if classes == 2:
            if probs.shape[1] > 1:            
                probs = probs[:, 1]
            fp_rate, tp_rate, thresholds = roc_curve(Y_test, probs)
            viz = {'fp_threshold': fp_rate, 'tp_threshold': tp_rate}
            results['viz'] = viz
            preds = np.zeros(len(probs))
            preds[probs.squeeze() > .5] = 1.
        else:
            preds = probs.argmax(axis=-1)
        
        self.cm.append(confusion_matrix(Y_test, preds))

        return results

    def results_reports(self, res):
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

            if self.__class__.__name__ == 'CrossSubject':
                self.predictions = np.array(self.predictions).reshape(
                    (folds, examples, dim))
            elif self.__class__.__name__ == 'SingleSubject':
                self.predictions = np.array(self.predictions).reshape(
                    (self.n_subjects, folds, examples, dim))

        res = aggregate_results(res)
        res = self._update_results(res)
        return res

    def save_model(self, folderpath=None, modelformat='TF'):
        """Save trained model in HDF5 format or SavedModel TF format
        Uses the built-in save method in Keras Model object.
        model name will be: folderpath/modelname_paradigm_dataset.h5

        Parameters
        ----------
        folderpath : str
            folder location where the model will be saved
            default : 'aawedha/trained
        """
        device = self._get_device()
        if not os.path.isdir('trained'):
            os.mkdir('trained')
        if not folderpath:
            folderpath = 'trained/'

        # filepath = folderpath + '/' + '_'.join([self.model.name, prdg, dt, '.h5'])
        prdg = self.dataset.paradigm.title
        dt = self.dataset.title
        filepath = os.path.join(folderpath, '_'.join([self.model.name, prdg, dt]))
        if self.engine == "keras":
            if modelformat == 'h5' or device == 'TPU':
                # due to cloud TPUs restrictions, we force
                # model saving to H5 format. used for long
                # benchmarking evaluations
                filepath = f"{filepath}.h5"
        else:
            filepath = f"{filepath}.pth" # pytorch model
        self.model.save(filepath)

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
        """
        if model_config:
            self.set_config(model_config)
        
        self.model = model
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
        """
        self.model_config = model_config

    def log_experiment(self):
        """Write in logger, evaluation information after completing a fold
        Message format :
        date-time-logger_name-logging_level-{fold|subject}-{ACC|AUC}-performance
        """
        s = ['train', 'val', 'test']
        
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
            chk = CheckPoint.load_checkpoint()

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
                self.best_kept = chk.best_kept
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
        # Keras models : 
            # layer 0 : Normalization
            # layer 1 : Reshape
            # layer 2 : Model
        # self.model.layers[2].set_weights(self.initial_weights)

    def _equal_subjects(self):
        """Test whether dataset's train_epochs and test_epochs has same number
        of subjects

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
        batch, ep, clbs, aug = self._get_fit_configs()

        device = self._get_device()
        '''
        if device != 'CPU':
            X_train, X_test, X_val = self._transpose_split(
                [X_train, X_test, X_val])
        '''
        #
        self.reset_weights()
        if self.normalize:
            X_train = self._normalize(X_train)
            
        # if label_smoothing
        if not aug:
            Y_train, Y_val, Y_test = self._to_categorical(Y_train, Y_val, Y_test)

        if X_val is None:
            val = None
        else:
            if self.engine == "pytorch" and self.normalize:
                X_val = self.model.normalize(X_val)
            val = (X_val, Y_val)

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
        
        if aug:
            X_train, val = build_mixup_dataset(X_train, Y_train, X_val, Y_val, aug, batch, self.engine)
            if isinstance(Y_test, np.ndarray):      
                Y_test = labels_to_categorical(Y_test)
            #
            history = self.model.fit(X_train,
                                 epochs=ep,
                                 steps_per_epoch=spe,
                                 verbose=self.verbose,
                                 validation_data=val,
                                 class_weight=cws,
                                 callbacks=clbs)
        else:
            history = self.model.fit(x=X_train, y=Y_train,
                                 batch_size=batch,
                                 epochs=ep,
                                 steps_per_epoch=spe,
                                 verbose=self.verbose,
                                 validation_data=val,
                                 class_weight=cws,
                                 callbacks=clbs)
        
        if isinstance(X_test, np.ndarray):
            if self.engine == "pytorch" and self.normalize:
                    X_test = self.model.normalize(X_test)
            probs = self.model.predict(X_test)
            perf = self.model.evaluate(X_test, Y_test, batch_size=batch, 
                                        return_dict=True, verbose=0)
        
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
            if 'metrics' in self.model_config['compile']:
                metrics = self.model_config['compile']['metrics']
            else:
                metrics = self._get_metrics()
            if 'loss' in self.model_config['compile']:
                khsara = self.model_config['compile']['loss']
            else:
                khsara = self._get_loss()
            if 'optimizer' in self.model_config['compile']:
                optimizer = self.model_config['compile']['optimizer']
                if isinstance(optimizer, str) or isinstance(optimizer, list):
                    if self.engine is "keras":
                        opt_lib = optimizer_lib(optimizer)
                        if opt_lib != 'builtin':
                            optimizer = get_optimizer(optimizer, opt_lib)
            else:
                optimizer = 'adam'
        else:
            khsara, optimizer, metrics = self._default_compile()
            # set config for checkpoint use            

        return khsara, optimizer, metrics

    def _default_compile(self):
        """Default model compile configuration, used when no compile
        config is passed to evaluation.
        """
        khsara = self._get_loss()
        if self._get_device() != 'TPU':
            metrics = self._get_metrics()
        else:
            metrics = []
        optimizer = "adam" if self.engine == "Keras" else "Adam"
        return khsara, optimizer, metrics

    def _normalize(self, X_train):
        """Adapt normalization layer if it's inside the model
        Parameters
        ----------
        X_train : ndarray
            training data n_samples x channels x samples
        """
        if self.engine == "keras":
            for layer in self.model.layers:
                if type(layer).__name__ is "Normalization":
                    layer.adapt(X_train)
        else:
            X_train = self.model.set_scale(X_train)
        return X_train

    def _get_fit_configs(self):
        """Returns fit configurations as tuple

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
            aug = None
            if 'augment' in self.model_config['fit']: 
                aug = self.model_config['fit']['augment']
            # format = self.model_config['fit']['format']
        else:
            batch = 64
            ep = 300
            clbks = []
            aug = None
            # format = 'channels_first'
        
        # K.set_image_data_format(format)
        
        if self.debug:
            # logdir = os.path.join("aawedha/debug", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            debug_dir = os.path.join("aawedha", "debug")
            self.log_dir = os.path.join(debug_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            if not os.path.isdir(self.log_dir):
                os.mkdir(self.log_dir)
            clbks.append(tf.keras.callbacks.TensorBoard(self.log_dir))
        
        return batch, ep, clbks, aug #, format

    def _get_model_configs_info(self):
        """Construct a logging message to be added to logger at evaluation beginning

        the message details the models configuration used for model

        Returns
        -------
        model_config : str
            model's configuration
        """
        khsara, opt, mets = self._get_compile_configs()
        batch, ep, clbs, aug = self._get_fit_configs()
        model_config = f' Loss: {khsara} | Optimizer: {opt} | \
                            metrics: {mets} | batch_size: {batch} | \
                            epochs: {ep} | \
                            callbacks: {clbs} \
                            augment : {aug}'
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
            '''
            metrics = ['accuracy',
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.TruePositives(name='tp'),
                        tf.keras.metrics.FalsePositives(name='fp'),
                        tf.keras.metrics.TrueNegatives(name='tn'),
                        tf.keras.metrics.FalseNegatives(name='fn'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')
                        ]
            '''
            metrics = metrics_by_lib(self.engine)
        else:
            metrics = ['accuracy']

        return metrics

    def _get_loss(self):
        """Get default suitable Loss name according to the number of classes in dataset.
        
        - Binary Classification : binary_crossentropy
        - Multi Class : sparse_categorical_crossentropy | categorical_crossentropy

        Returns
        -------
        str : name of Loss
        """
        classes = self._get_classes()
        if classes == 2:
            return 'binary_crossentropy'
        elif 'fit' in self.model_config:
            if 'augment' in self.model_config['fit']:
                return 'categorical_crossentropy'
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

    def _to_categorical(self, Y_train, Y_val, Y_test):
        """Convert numerical data labels into categorical vector labels.

        Parameters
        ----------
        Y_train : 1d array (N_training)
            training data labels
        Y_val : 1d array (N_validation)
            validation data labels
        Y_test : 1d array (N_test)
            test data labels

        Returns
        -------
        Y_train: ndarray (N_training, N_classes)
            training data labels in categorical format.
        Y_val: ndarray (N_validation, N_classes)
            validation data labels in categorical format.
        Y_test: ndarray (N_test, N_classes)
            test data labels in categorical format.
        """
        convert_label = False
        khsara, _, _ = self._get_compile_configs()
             
        if self.engine == 'keras' and type(khsara) != str:
            loss_config = khsara.get_config()
            if khsara.name != 'sparse_categorical_crossentropy' and 'label_smoothing' in loss_config:
                if loss_config['label_smoothing'] != 0.0:
                    convert_label = True              
                  
        if convert_label:
            Y_train = labels_to_categorical(Y_train)
            if isinstance(Y_test, np.ndarray):
                Y_test = labels_to_categorical(Y_test)
            if isinstance(Y_val, np.ndarray):
                Y_val = labels_to_categorical(Y_val)
        
        return Y_train, Y_val, Y_test

    @staticmethod
    def _create_split(X_train, X_val, X_test, Y_train, Y_val, Y_test):
        """gather data arrays in a dict

        Parameters
        ----------
        X_train : ndarray (N_training, channels, samples)
            training data
        X_val : ndarray (N_validation, channels, samples)
            validation data
        X_test : ndarray (N_test, channels, samples)
            test data
        Y_train : 1d array
            training data labels
        Y_val : 1d array
            validation data labels
        Y_test : 1d array 
            test data labels

        Returns
        -------
        dict
            evaluation data split dictionary where the key is the array's name.
        """
        split = {}
        split['X_test'] = None
        split['X_val'] = None
        split['X_train'] = X_train if X_train.dtype is np.float32 else X_train.astype(np.float32)
        split['Y_train'] = Y_train
        if isinstance(X_test, np.ndarray):
            split['X_test'] = X_test if X_test.dtype is np.float32 else X_test.astype(np.float32)
        split['Y_test'] = Y_test
        if isinstance(X_val, np.ndarray):
            split['X_val'] = X_val if X_val.dtype is np.float32 else X_val.astype(np.float32)
        split['Y_val'] = Y_val
        return split
    
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
            devices = [dev.device_type for dev in tf.config.get_visible_devices()]
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

    def _post_operations(self, savecsv=False, csvfolder=None):
        """Log results and save them as a pandas DataFrame 

        Parameters
        ----------
        savecsv : bool, optional
            if True, save results as a pandas DataFrame in a csv file. By default False
        csvfolder : str, optional
            folder path where to save results, by default None
        """
        if self.log:
            self._log_results()

        if savecsv:
            if self.results:
                self._savecsv(csvfolder)
        
    def _savecsv(self, folder=None):
        """Save evaluation results in a CSV file as Pandas DataFrame

        Parameters
        ----------
        folder : str
            results files will be stored inside folder, if None, a default folder inside aawedha is used.
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
        
        if evl == 'CrossSubject' or evl == 'CrossSet':
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
