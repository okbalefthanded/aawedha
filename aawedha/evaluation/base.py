
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
from aawedha.utils.utils import log
from aawedha.evaluation.checkpoint import CheckPoint
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import datetime
import random
import pickle
import abc
import os


class Evaluation(object):
    '''Base class for evaluations

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
            configuration for data partioning into train/validation/test subset
            default a 3 integers list of:
                In case of a single dataset without an independet Test set
                - (folds/L, folds/M, folds/N) L+M+N = total trials in dataset for SingleSubject evaluation
                - (L_subjects, M_subjects, N_subjects) L+M+N = total subjects in dataset for CrossSubject evaluation
            a 2 integers list of:
                In case of a dataset with an independet Test set
                - (folds/L, folds/M) L+M = T total trials in dataset for SingleSubject evaluation
                - (L_subjects, M_subjects) L+M = S total subjects in dataset for CrossSubject evaluation

        folds : a list of 3 1d numpy array
            indices of trials(SingleSubject evaluation)/subjects(CrossSubjects evaluation) for each fold

        verbose : int
            level of verbosity for model at fit method , 0 : silent, 1 : progress bar, 2 : one line per epoch

        lg : bool
            if True uses logger to log experiment configurations and results, default False

        predictions : ndarray of predictions
            models output for each example on the dataset :
                - SingleSubject evaluaion : subjects x folds x Trials x dim
                - CrossSubject evaluation : folds x Trials x dim

        cm : list
            confusion matrix per fold

        results : dict of evaluation results compiled from models performance on the dataset
            - 'acc' : 2d array : Accuracy for each subjects on each folds (subjects x folds)
            - 'acc_mean' : double : Accuracy mean over all subjects and folds
            - 'acc_mean_per_fold' : 1d array : Accuracy mean per fold over all subjects
            - 'acc_mean_per_subj' : 1d array : Accuracy mean per Subject over all folds [only for SingleSubject evaluation]
            For binary class tasks :
            - 'auc' : 2d array : AUC for each subjects on each folds (subjects x folds)
            - 'auc_mean' : double :  AUC mean over all subjects and folds
            - 'auc_mean_per_fold' :  1d array : AUC mean per fold over all subjects          
            - 'auc_mean_per_subj' :  AUC mean per Subject over all folds
                [only for SingleSubject evaluation]
            - 'tpr' : 1d array : True posititves rate
            - 'fpr' : 1d array : False posititves rate

        n_subjects : int
            number of subjects in dataset if the train and test set have same
            subjects, the sum of both, otherwise.

        model_history : list
            Keras history callbacks

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

        model_compiled : bool, flag for model state
            default : False

    '''
    def __init__(self, dataset=None, model=None, partition=[], folds=[],
                 verbose=2, lg=False):
        '''
        '''
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
            now = datetime.datetime.now().strftime('%c').replace(' ', '_')
            f = 'aawedha/logs/'+'_'.join([self.__class__.__name__,
                                         dataset.title, now, '.log'])
            self.logger = log(fname=f, logger_name='eval_log')
        else:
            self.logger = None
        self.model_compiled = False
        self.model_config = {}
        self.current = None

    @abc.abstractmethod
    def generate_split(self, nfolds):
        '''Generate train/validation/test split
            Overriden in each type of evaluation : SingleSubject | CrossSubject
        '''
        pass

    @abc.abstractmethod
    def run_evaluation(self):
        '''Main evaluation process 
            Overriden in each type of evaluation : SingleSubject | CrossSubject
        '''
        pass

    def resume(self, run=False):
        '''Resume evaluation from where it was interrupted

        Parameters
        ----------
        None

        Returns
        -------
        no value
        '''

        '''
        file_name = 'aawedha/checkpoints/current_CheckPoint.pkl'
        if os.path.exists(file_name):
            f = open(file_name, 'rb')
            chkpoint = pickle.load(f)
        else:
            raise FileNotFoundError
        f.close()
        '''
        chk = self.reset(chkpoint=True)
        if run:
            self.run_evaluation(pointer=chk, check=True)
        return self

    def set_dataset(self, dt=None):
        '''Instantiate dataset with dt
        
        Parameters
        ----------
        dt : dataset instance
            a dataset object
        
        Returns
        -------
        no value
        '''
        self.dataset = dt

    def measure_performance(self, Y_test, probs):
        '''Measure model performance on dataset

        Calculates model performance on metrics and sets Confusion Matrix for each fold

        Parameters
        ----------
        Y_test : 2d array (n_examples x n_classes) 
            true class labels in Tensorflow format

        probs : 2d array (n_examples x n_classes)
            model output predictions as probability of belonging to a class   

        Returns
        -------
            acc : float
                accuracy value of model per fold
            auc_score : float
                AUC value of model per fold (only for Binary class tasks)
            fp_rate : 1d array 
                increasing false positive rate
            tp_rate : 1d array
                increasing true positive rate
        '''
        self.predictions.append(probs)  # ()
        preds = probs.argmax(axis=-1)
        y_true = Y_test.argmax(axis=-1)
        classes = Y_test.shape[1]
        acc = np.mean(preds == y_true)

        self.cm.append(confusion_matrix(Y_test.argmax(axis=-1), preds))

        if classes == 2:
            fp_rate, tp_rate, thresholds = roc_curve(y_true, probs[:, 1])
            auc_score = auc(fp_rate, tp_rate)
            return acc.item(), auc_score.item(), fp_rate, tp_rate
        else:
            return acc.item()

    def results_reports(self, res, tfpr={}):
        '''Collects evaluation results on a single dict 

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
        results : dict of evaluation results compiled from models performance on the dataset
            - 'acc' : 2d array : Accuracy for each subjects on each folds (subjects x folds)
            - 'acc_mean' : double : Accuracy mean over all subjects and folds
            - 'acc_mean_per_fold' : 1d array : Accuracy mean per fold over all subjects
            - 'acc_mean_per_subj' : 1d array : Accuracy mean per Subject over all folds [only for SingleSubject evaluation]
            For binary class tasks :
            - 'auc' : 2d array : AUC for each subjects on each folds (subjects x folds)
            - 'auc_mean' : double :  AUC mean over all subjects and folds
            - 'auc_mean_per_fold' :  1d array : AUC mean per fold over all subjects          
            - 'auc_mean_per_subj' :  AUC mean per Subject over all folds [only for SingleSubject evaluation]
            - 'tpr' : 1d array : True posititves rate
            - 'fpr' : 1d array : False posititves rate
        '''
 
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
        #
        results = {}
        #
        if tfpr:
            # res : (metric, subjects, folds)
            # means = res.mean(axis=-1) # mean across folds
            r1 = np.array(res['acc'])
            r2 = np.array(res['auc'])
            results['auc'] = r2
            results['acc'] = r1
            results['acc_mean_per_fold'] = r1.mean(axis=0)
            results['auc_mean_per_fold'] = r2.mean(axis=0)
            results['acc_mean'] = r1.mean()
            results['auc_mean'] = r2.mean()
            #
            if self.__class__.__name__ == 'SignleSubject':
                results['acc_mean_per_subj'] = r1.mean(axis=1)
                results['auc_mean_per_subj'] = r2.mean(axis=1)
            #
            results['fpr'] = tfpr['fp']
            results['tpr'] = tfpr['tp']
        else:
            # res : (subjects, folds)
            results['acc'] = res
            # mean across folds
            results['acc_mean_per_fold'] = res.mean(axis=0)
            # mean across subjects and folds
            if self.__class__.__name__ == 'SignleSubject':
                results['acc_mean_per_subj'] = res.mean(axis=1)
            results['acc_mean'] = res.mean()

        return results

    def get_folds(self, nfolds, population, tr, vl, ts, exclude_subj=True):
        '''Generate train/validation/tes folds following Shuffle split strategy

        Parameters
        ----------
        nfolds : int
            number of folds to generate
        population : int
            number of total trials/subjects available
        tr : int
            number of trials/subjects to include in train folds
        vl : int
            number of trials/subjects to include in validation folds
        ts : int
            number of trials/subjects to include in test folds

        exclude_subj : bool (only in CrossSubject evaluation)
            if True the target subject data will be excluded from train fold

        Returns
        -------
        folds : list
        '''
        folds = []
        if hasattr(self.dataset, 'test_epochs'):
            if self.__class__.__name__ == 'CrossSubject':
                # independent test set
                # list : nfolds : [nsubjects_train] [nsubjects_val]
                for subj in range(self.n_subjects):
                    selection = np.arange(0, self.n_subjects)
                    if exclude_subj:
                        # fully cross-subject, no subject train data in fold
                        selection = np.delete(selection, subj)
                    for fold in range(nfolds):
                        np.random.shuffle(selection)
                        folds.append([np.array(selection[:tr]),
                                      np.array(selection[tr:tr + vl]),
                                      np.array([subj])
                                      ])
            elif self.__class__.__name__ == 'SingleSubject':
                # generate folds for test set from one set
                pop = population
                t = tr
                v = vl
                s = ts
                for f in range(nfolds):
                    if isinstance(population, np.ndarray):
                        # inconsistent numbers of trials among subjects
                        sbj = []
                        for subj in range(self.n_subjects):
                            pop = population[subj]
                            t = tr[subj]
                            v = vl[subj]
                            s = ts[subj]
                            tmp = np.array(random.sample(range(pop), pop))
                            sbj.append([tmp[:t], tmp[t:t + v], tmp[-s:]])
                        folds.append(sbj)
                    else:
                        # same numbers of trials for all subjects
                        tmp = np.array(random.sample(range(pop), pop))
                        folds.append([tmp[:t], tmp[t:t + v], tmp[-s:]])
        else:
            # generate folds for test set from one set
            if self.__class__.__name__ != 'SingleSubject':
                for subj in range(self.n_subjects):
                    # exclude subject per default
                    selection = np.arange(0, self.n_subjects)
                    # fully cross-subject, no subject train data in fold
                    selection = np.delete(selection, subj)
                    for fold in range(nfolds):
                        np.random.shuffle(selection)
                        folds.append([np.array(selection[:tr]),
                                      np.array(selection[tr:tr + vl]),
                                      np.array([subj])
                                      ])
            else:
                for _ in range(nfolds):
                    tmp = np.array(random.sample(range(population), population))
                    folds.append([tmp[:tr], tmp[tr:tr + vl], tmp[-ts:]])
        #
        return folds

    def fit_scale(self, X):
        '''Estimate mean and standard deviation from train set
        for normalization.
        train data is normalized afterwards

        Parameters
        ----------
        X : nd array (trials, kernels, samples, channels)
            training data

        Returns
        -------
        X :  nd array (trials, kernels, samples, channels)
            normalized training data
        mu : nd array (1, kernels, samples, channels)
            mean over all trials
        sigma : nd array (1, kernels, samples, channels)
            standard deviation over all trials
        '''
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        X = np.subtract(X, mu[None, :, :])
        X = np.divide(X, sigma[None, :, :])
        return X, mu, sigma

    def transform_scale(self, X, mu, sigma):
        '''Apply normalization on validation/test data using estimated
        mean and std from fit_scale method

        Parameters
        ----------
        X :  nd array (trials, kernels, samples, channels)
            normalized training data
        mu : nd array (1, kernels, samples, channels)
            mean over all trials
        sigma : nd array (1, kernels, samples, channels)
            standard deviation over all trials

        Returns
        -------
        X :  nd array (trials, kernels, samples, channels)
            normalized data
        '''
        X = np.subtract(X, mu[None, :, :])
        X = np.divide(X, sigma[None, :, :])
        return X

    def class_weights(self, y):
        '''Calculates inverse of ratio of class' examples in train dataset
        used to re-weight loss function in case of imbalanced classes in data

        Parameters
        ----------
        y : 1d array of int
            true labels

        Returns
        -------
        cl_weights : dict of (int : float)
            class_weight : class, 1 for each class if data classes are balanced

        '''
        cl_weights = {}
        classes = np.unique(y)
        n_perclass = [np.sum(y == cl) for cl in classes]
        n_samples = np.sum(n_perclass)
        ws = np.array([np.ceil(n_samples / cl).astype(int)
                       for cl in n_perclass])
        if np.unique(ws).size == 1:
            # balanced classes
            cl_weights = {cl: 1 for cl in classes}
        else:
            # unbalanced classes
            if classes.size == 2:
                cl_weights = {classes[ws == ws.max()].item(
                ): ws.max(), classes[ws < ws.max()].item(): 1}
            else:
                cl_weights = {cl: ws[idx] for idx, cl in enumerate(classes)}
        return cl_weights

    def labels_to_categorical(self, y):
        '''Convert numerical labels to categorical

        Parameters
        ----------
        y : 1d array
            true labels array in numerical format : 1,2,3,

        Returns
        -------
        y : 2d array (n_examples x n_classes)
            true labels in categorical format : example [1,0,0]
        '''
        classes = np.unique(y)
        if np.isin(0, classes):
            y = to_categorical(y)
        else:
            y = to_categorical(y - 1)
        return y

    def save_model(self, folderpath=None):
        '''Save trained model in HDF5 format
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
        '''

        if not os.path.isdir('trained'):
            os.mkdir('trained')
        if not folderpath:
            folderpath = 'trained'
        prdg = self.dataset.paradigm.title
        dt = self.dataset.title
        filepath = folderpath + '/' + '_'.join([self.model.name, prdg, dt, '.h5'])
        self.model.save(filepath)

    def set_model(self, model=None, model_config={}):
        '''Assign Model and model_config

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
        '''
        self.model = model
        if model_config:
            self.model_config = model_config

    def set_config(self, model_config):
        '''
        '''
        self.model_config = model_config

    def log_experiment(self):
        '''Write in logger, evaluation information after completing a fold
        Message format : date-time-logger_name-logging_level-{fold|subject}-{ACC|AUC}-performance

        Parameters
        ----------
        None

        Returns
        -------
        no value
        '''
        s = ['train', 'val', 'test']
        data = f' Dataset: {self.dataset.title}'
        if isinstance(self.dataset.epochs, list):
            duration = f' epoch duration:{self.dataset.epochs[0].shape[0]/self.dataset.fs} sec'
        else:
            duration = f' epoch duration:{self.dataset.epochs.shape[1]/self.dataset.fs} sec'
       
        prt = 'Subjects partition '+', '.join(f'{s[i], self.partition[i]}' for i in range(len(self.partition)))
        model = f'Model: {self.model.name}'
        model_config = f'Model config: {self._get_model_configs_info()}'
        exp_info = ' '.join([data, duration, prt, model, model_config])
        self.logger.debug(exp_info)

    def reset(self, chkpoint=False):
        '''Reset Attributes and results for a future evaluation with
            different model and same partition and folds

        if chkpoint is passed, Evaluation attributes will be set to 
        the state at where the evaulation was interrupted, this is used
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
        '''
        chk = None
        if chkpoint:
            chk = self._load_checkpoint()

        # if chkpoint:
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

        else:
            self.model = None
            self.predictions = []
            self.cm = []  # confusion matrix per fold
            self.results = {}
            self.model_history = {}
            self.model_compiled = False
            self.model_config = {}

        return chk

    def _equale_subjects(self):
        '''Test whether dataset's train_epochs and test_epochs has same number
        of subjects

        Parameters
        ----------
        None

        Returns
        -------
        bool : True if number of subjects in training data equals the number of
        subjects in test data False otherwise
        '''

        ts = 0
        tr = len(self.dataset.epochs)
        if hasattr(self.dataset, 'test_epochs'):
            ts = len(self.dataset.test_epochs)
        return tr == ts

    def _get_n_subjects(self):
        '''Return number of subjects in dataset

        Parameters
        ----------
        None

        Returns
        -------
        int : number of subjects if train/test subjects is the same
                their sum otherwise
        '''
        if self.dataset:
            ts = len(self.dataset.test_epochs) if hasattr(self.dataset, 'test_epochs') else 0
            if self._equale_subjects():
                return len(self.dataset.epochs)
            else:
                return len(self.dataset.epochs) + ts
        else:
            return 0

    def _compile_model(self):
        '''Compile model using speficied model_config, default values otherwise

        Sets model_compiled attribute to true after successful model compilation

        Parameters
        ----------
        None

        Returns
        -------
        no value

        '''
        if not self.model_config:
            # some nice default configs
            khsara, opt, mets = self._get_compile_configs()
        else:
            khsara = self.model_config['loss']
            opt = self.model_config['optimizer']
            mets = self.model_config['metrics']

        self.model.compile(loss=khsara,
                           optimizer=opt,
                           metrics=mets
                           )
        self.model_compiled = True

    def _eval_model(self, X_train, Y_train, X_val, Y_val, X_test, cws):
        '''Train model on train/validation data and predict its output on test data

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
            model's output on test data as probabilites of belonging to each class

        '''
        batch, ep, clbs = self._get_fit_configs()
        history = self.model.fit(X_train, Y_train,
                                 batch_size=batch, epochs=ep,
                                 verbose=self.verbose,
                                 validation_data=(X_val, Y_val),
                                 class_weight=cws,
                                 callbacks=clbs)
        probs = self.model.predict(X_test)
        return history, probs

    def _get_compile_configs(self):
        '''Returns default model compile configurations as tuple

        Parameters
        ----------
        None

        Returns
        -------
        khsara (loss in Arabic): str
            loss function optimized during training
        opt : str
            optimizer
        mets : list : str| keras metrics
            metrics
        '''
        classes = self.dataset.get_n_classes()
        if self.model_config:
            mets = self.model_config['metrics']
            khsara = self.model_config['loss']
            opt = self.model_config['optimizer']
        else:
            mets = ['accuracy']
            if classes == 2:
                khsara = 'binary_crossentropy'
                mets.append(AUC())
            else:
                khsara = 'categorical_crossentropy'
            opt = 'adam'
        return khsara, opt, mets

    def _get_fit_configs(self):
        '''Returns fit configurations as tuple

        Parameters
        ----------
        None

        Returns
        -------
        batch : int
            batch size
        ep : int
            epochs number
        clbs : list
            keras callbacks added to be watched during training
        '''
        if self.model_config:
            batch = self.model_config['batch']
            ep = self.model_config['epochs']
            clbs = self.model_config['callbacks']
        else:
            batch = 64
            ep = 300
            clbs = []
        return batch, ep, clbs

    def _get_model_configs_info(self):
        '''Construct a logging message to be added to logger at evaluation beginning

        the message details the models configuration used for model

        Parameters
        ----------
        None

        Returns
        -------
        model_config : str
            model's configuation
        '''
        khsara, opt, mets = self._get_compile_configs()
        batch, ep, clbs = self._get_fit_configs()
        model_config = f' Loss: {khsara} | Optimizer: {opt} | metrics: {mets} | batch_size: {batch} | epochs: {ep} | callbacks: {clbs}'
        return model_config

    def _assert_partiton(self, excl=False):
        '''Assert if partition to be used do not surpass number of subjects available
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
        '''
        subjects = self._get_n_subjects()
        prt = np.sum(self.partition)
        return subjects < prt

    def _load_checkpoint(self):
        '''
        '''
        file_name = 'aawedha/checkpoints/current_CheckPoint.pkl'
        if os.path.exists(file_name):
            f = open(file_name, 'rb')
            chkpoint = pickle.load(f)
        else:
            raise FileNotFoundError
        f.close()
        return chkpoint
