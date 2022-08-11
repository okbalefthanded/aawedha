from aawedha.evaluation.evaluation_utils import labels_to_categorical
from aawedha.evaluation.evaluation_utils import measure_performance
from aawedha.evaluation.evaluation_utils import class_weights
from aawedha.evaluation.mixup import build_mixup_dataset
from aawedha.evaluation.checkpoint import CheckPoint
from aawedha.models.utils_models import load_model
from aawedha.evaluation.settings import Settings
from aawedha.utils.utils import make_folders
from aawedha.utils.utils import get_gpu_name
from aawedha.models.base_model import Model
from aawedha.utils.utils import get_device
from aawedha.evaluation.score import Score
from aawedha.utils.utils import time_now
from aawedha.utils.logger import Logger
from aawedha.io.base import DataSet
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import abc
import os


class Evaluation:
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

        settings.partition : list of 2 or 3 integers
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
                 verbose=2, log=False, engine="pytorch", normalize=True, 
                 debug=False):
        """
        """
        self.dataset = dataset
        self.settings = Settings(partition, folds, engine, verbose, 0, debug)        
        self.learner = Model(model, normalize=normalize)
        self.score = Score()      
        self.n_subjects = self._get_n_subjects()
        self.log = log
        self.logger = None        
        if self.log:
            self.logger = self._init_logger()           
        self.log_dir = None 

    def __str__(self):
        name = self.__class__.__name__
        model = self.learner.model.name if self.learner else 'NotSet'
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
        device = get_device(self.learner.config)
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
            self.learner.config = model_config
            self.settings.fit_config = model_config['fit']

        self.learner.model = model
        self.learner.name = model.name
        self.learner.set_type()
        self.reset_weights()    

    def save_model(self, folderpath=None, filepath=None, modelformat='TF'):
        """Save trained model in HDF5 format or SavedModel TF format
        Uses the built-in save method in Keras Model object.
        model name will be: folderpath/modelname_paradigm_dataset.h5

        Parameters
        ----------
        folderpath : str
            folder location where the model will be saved
            default : 'aawedha/trained

        filepath : str
            full filepath and model name
            default : None

        modelformat : str
            model's file format 
            TF:  Tensorflow SavedFormat
            H5:  Keras h5py format
            PTH: PyTorch format
        """
        device = get_device(self.learner.config)
        if not os.path.isdir('trained'):
            os.mkdir('trained')
        if not folderpath:
            folderpath = 'trained/'

        # filepath = folderpath + '/' + '_'.join([self.model.name, prdg, dt, '.h5'])        
        if not filepath:
            prdg = self.dataset.paradigm.title
            dt = self.dataset.title
            filepath = os.path.join(folderpath, '_'.join([self.learner.name, prdg, dt]))
        if self.settings.engine == "keras":
            if modelformat == 'h5' or device == 'TPU':
                # due to cloud TPUs restrictions, we force
                # model saving to H5 format. used for long
                # benchmarking evaluations
                filepath = f"{filepath}.h5"
        else:
            filepath = f"{filepath}.pth" # pytorch model
        self.learner.save(filepath)            

    def log_experiment(self):
        """Write in logger, evaluation information after completing a fold
        Message format :
        date-time-logger_name-logging_level-{fold|subject}-{ACC|AUC}-performance
        """
        s = ['train', 'val', 'test']
        
        # data, duration, data_shape = self._dataset_info()
        if self.dataset:
            data, duration, data_shape = self.dataset.info()
        else:
            data, duration, data_shape = '', '0', '[]'

        prt = 'Subjects partition ' + \
              ', '.join(f'{s[i], self.settings.partition[i]}' for i in range(
                  len(self.settings.partition)))
        model = f'Model: {self.learner.name}'
        model_config = f'Model config: {self._get_model_configs_info()}'
        device = get_device(self.learner.config)
        if device == 'GPU':
            compute_engine = get_gpu_name()
        else:
            compute_engine = device

        mode = ''
        if hasattr(self, 'mode'):
            mode = f"Cross Set Mode: {self.mode}"

        exp_info = ' '.join([data, duration, data_shape, prt, self.settings.engine, model,
                             model_config, compute_engine, mode])
        # self.logger.debug(exp_info)
        self.logger.log(exp_info)

    def reset(self, chkpoint=False):
        """Reset Attributes and results for a future evaluation with
            different model and same settings.partition and folds

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
            self.learner = Model(load_model(chk.check_path), 
                                  compiled=chk.learner.compiled, 
                                  weights=chk.learner.initial_weights, 
                                  config=chk.learner.config, 
                                  history=chk.learner.history, 
                                  normalize=chk.learner.do_normalize, 
                                  name=chk.learner.name, 
                                  model_type=chk.learner.type)
            self.settings = chk.settings
            self.settings.current = chk.current
            self.score.results = chk.rets            
            self.log = chk.log
            self.logger = Logger(fname=chk.logger, logger_name='eval_log')
            
            if hasattr(chk, 'mode'):
                self.mode = chk.mode
                self.best_kept = chk.best_kept
        else:        
            self.learner = Model(model=None)
            self.score = Score()  

        return chk

    def reset_weights(self):
        """reset model's weights to initial state (model's creation state)
        """
        self.learner.reset_weights()

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
        device = get_device(self.learner.config)
        classes = self._get_classes()
        self.learner.compile(device, classes)
        self.score.build(self.learner.model.metrics_names)

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
        device = get_device(self.learner.config)
        '''
        if device != 'CPU':
            X_train, X_test, X_val = self._transpose_split(
                [X_train, X_test, X_val])
        '''
        #
        self.reset_weights()
        if self.learner.do_normalize:
            X_train = self.learner.fit_normalize(X_train)
 
        # if label_smoothing
        if not aug:
            Y_train, Y_val, Y_test = self._to_categorical(Y_train, Y_val, Y_test)

        if X_val is None:
            val = None
        else:            
            X_val = self.learner.normalize(X_val)
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
            X_train, val = build_mixup_dataset(X_train, Y_train, X_val, Y_val, aug, 
                                               batch, self.settings.engine)
            if isinstance(Y_test, np.ndarray):      
                Y_test = labels_to_categorical(Y_test)
            #
            Y_train = None
        
        history = self.learner.fit(x=X_train, y=Y_train,
                                 batch_size=batch,
                                 epochs=ep,
                                 steps_per_epoch=spe,
                                 verbose=self.settings.verbose,
                                 validation_data=val,
                                 class_weight=cws,
                                 callbacks=clbs)
        
        if isinstance(X_test, np.ndarray):
            probs = self.learner.predict(X_test)
            perf  = self.learner.evaluate(X_test, Y_test, batch_size=batch, 
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
        model_history, probs, perf = self._eval_model(X_train, Y_train,
                                                           X_val, Y_val,
                                                           X_test, Y_test,
                                                           cws)
        
        self.learner.history.append(model_history)

        if isinstance(X_test, np.ndarray):            
            rets = measure_performance(Y_test, probs, perf, self.learner.model.metrics_names)
        return rets        

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
        batch, ep, clbks, aug = self.settings.get_fit_configs()
        if self.settings.debug:
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
        device = get_device(self.learner.config)
        classes = self._get_classes()
        khsara, opt, mets, schedule = self.learner.get_compile_configs(device, classes)
        batch, ep, clbs, aug = self._get_fit_configs()
        model_config = f' Loss: {khsara} | Optimizer: {opt} | \
                            metrics: {mets} | batch_size: {batch} | \
                            epochs: {ep} | \
                            Schedule : {schedule} \
                            callbacks: {clbs} \
                            augment : {aug}'
        return model_config

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
        device = get_device(self.learner.config)
        classes = self._get_classes()
        khsara, _, _, _ = self.learner.get_compile_configs(device, classes)
             
        if self.settings.engine == 'keras' and type(khsara) != str:
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
    
    def _assert_partition(self, excl=False):
        """Assert if settings.partition to be used do not surpass number of subjects available
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
        prt = np.sum(self.settings.partition)
        return subjects < prt

    def _has_val(self):
        """test if evaluation has settings.partition for validation phase

        Returns
        -------
        int :
           if 0 no validation, otherwise number of parts from dataset to use
            in validation
        """
        train = self.settings.partition[0]
        if len(self.settings.partition) == 3:
            test = self.settings.partition[2]
        else:
            test = 1

        return self._get_n_subjects() - train - test

    def _init_logger(self):
        if self.dataset:
            title = self.dataset.title
        else:
            title = ''
        if not os.path.isdir("aawedha/logs"):
                make_folders()
        dataset_folder = f"aawedha/logs/{title}/"
        now = datetime.datetime.now().strftime('%c').replace(' ', '_').replace(':', '_')
        if not os.path.isdir(dataset_folder):
            os.mkdir(dataset_folder)
        f = dataset_folder + '_'.join([self.__class__.__name__,
                                           title, now, '.log'])
        return Logger(fname=f, logger_name='eval_log')
    
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
        msg = f" Subj {op_ind+1} "
        msg += " ".join([f"{metric}: {v} " 
               for metric, v in op_results.items() 
               if metric not in ["probs", "confusion"]])

        if isinstance(self.learner.history[0], dict):
            epochs = len(self.learner.history[0]['history']['loss']) 
        else:
            epochs = self.learner.history[0].params['epochs'] # Keras History object
        msg += f" Training stopped at epoch: {epochs}"
        self.logger.log(msg)    
    
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
            self.logger.log_results(self.score)

        if savecsv:
            if self.score.results:
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

        metrics = list(self.score.results)
        [metrics.remove(elem) for elem in ["probs", "confusion"]]
        m = [met  for met in metrics if "mean" not in met]
        metrics = m
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
            metrics.remove("folds")
        elif evl == 'SingleSubject':
            metrics.remove("subjects")
            nfolds = len(self.score.results['accuracy'][0])
            # columns = [f'Fold {fld+1}' for fld, _ in enumerate(evl.folds)]
            columns = [f'Fold {fld+1}' for fld in range(nfolds)]

        columns.extend(['Avg', 'Std', 'Sem'])
        date = time_now()
        results = self.score.results
        for metric in metrics:
            acc = results[metric].round(3) 
            if acc.ndim == 1:
                acc_mean = acc
                std = results[metric].std().round(3)
                std = np.tile(std, len(rows) - 1)
            else:
                acc_mean = results[metric].mean(axis=1).round(3)
                std = results[metric].std(axis=1).round(3) 
            sem = np.round(std / np.sqrt(len(results[metric])), 3)
            values = np.column_stack((acc, acc_mean, std, sem))
            values = np.vstack((values, values.mean(axis=0).round(3)))
            df = pd.DataFrame(data=values, index=rows, columns=columns)
            df.index.name = f"{self.learner.model.name} / {metric}"
            fname = f"{folder}/{evl}_{dataset}_{metric}_{date}.csv"
            df.to_csv(fname, encoding='utf-8')
            print(f"Results saved as CSV in: {fname}")
