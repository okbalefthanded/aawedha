from aawedha.evaluation.base import Evaluation
from aawedha.evaluation.checkpoint import CheckPoint
from aawedha.utils.evaluation_utils import class_weights, labels_to_categorical
from aawedha.utils.evaluation_utils import fit_scale, transform_scale
from aawedha.analysis.utils import isfloat
import numpy as np


class CrossSet(Evaluation):
    """Cross Set Evaluation

    derived from base Evaluation class, performs cross set transfer learning

    Attributes
    ----------
    source : list
        list of datasets instances used to augment data in training.

    target : dataset instance
        dataset used for both training and evaluation

    Methods
    -------
    select_channels()

    select_trials()

    resample()

    generate_set()

    generate_split()

    run_evaluation()

    _is_selectable()

    _cross_set()

    _diff()

    _split_set()

    _flatten()

    _get_min_channels()
    """

    def __init__(self, source=[], target=[], mode='', partition=[],
                 verbose=2, lg=False):
        '''
        '''
        self.source = source
        self.target = target
        self.mode = mode
        super().__init__(partition=partition,
                         verbose=verbose, lg=lg)

    def __str__(self):
        name = self.__class__.__name__
        model = self.model.name if self.model else 'NotSet'
        source_info = '\n'.join([str(src) for src in self.source])
        info = (f'Type: {name}',
                f'Target Set: {self.target.info}'
                f'Source: {source_info}',
                f'Model: {model}')
        return '\n'.join(info)

    def select_channels(self, ds=[], chs=[], replacement={}):
        '''Select a subset of channels available among all datasets
        Parameters
        ----------
        ds : dataset instance
            a dataset to use in evaluation

        chs: list of str
            minimum channels the dataset will be resized from

        replacement: dict
            nearest electrodes in dataset to replace missing ones
            in chs

        Returns
        -------
        None
        '''
        channels = list(set(ds.ch_names).intersection(chs))

        if len(channels) == len(chs):
            # dataset contain subset of channels
            ds.select_channels(chs)
        else:
            # find and/or replace unmatched channels
            # must have replacement
            for (i, v) in enumerate(chs):
                for r in replacement:
                    if r == v:
                        chs[i] = replacement[r]
            ds.select_channels(chs)

    def select_trials(self, source=[]):
        """Keep trials in source datasets based on events from target
        dataset

        Parameters
        ----------
        source : dataset instance

        Returns
        -------
        no value
        """
        target_events = np.unique(self.target.events)

        d_source = self._diff(source.events)
        d_target = self._diff(target_events)
        v = np.min([d_source, d_target])

        # events = np.unique(ev.astype(float))
        # events = np.unique(ev)
        # labels = np.unique(self.target.y)
        # new_labels = {str(events[i]): labels[i] for i in range(events.size)}
        # new_labels = {events[i]: labels[i] for i in range(events.size)}

        # keys = self.target.events.flatten().tolist()
        # values = self.target.y.flatten().tolist()
        # new_labels = dict(zip(keys, values))

        new_labels = self.target.labels_to_dict()
        source.rearrange(target_events, v)
        source.update_labels(new_labels, v)
        # need to recover original indices of trials

    def resample(self):
        '''Resample all datasets (target and sources alike) used in evaluation
            to lowest frequency sampling among them.

        Parameters
        ----------
        no parameters

        Returns
        -------
        no value
        '''
        fs_all = [src.fs for src in self.source]
        fs_all.append(self.target.fs)
        fs_all = np.array(fs_all)
        min_fs = fs_all.min()

        for src in self.source:
            src.resample(min_fs)

        self.target.resample(min_fs)

    def generate_set(self, replacement={}):
        '''Unify datasets in a single format by keeping shared channels, trials
        and resampling.

        Parameters
        ----------
        replacement: dict
            nearest electrodes in dataset to replace missing ones
            in chs

        Returns
        -------
        no value
        '''
        #
        chs = self._get_min_channels()
        # ev = np.unique(self.target.events)

        # select channels and trials for source datasets
        for src in self.source:
            self.select_channels(src, chs)
            if not self._is_selectable(src):
                self.select_trials(src)

        if replacement:
            self.select_channels(self.target, chs, replacement)
        else:
            self.select_channels(self.target, chs)
        #
        self.resample()

    def generate_split(self, nfolds=30, excl=True, replacement=[]):
        '''Generate cross-validation folds following a cross-validation
        strategy from ShuffleSplit

        Parameters
        ----------
        nfolds : int
            number of cross-validation folds to generate
            default : 30

        excl : bool
            True, exclude target subject from evaluation to insure a complete
            cross-subject evaluation, False otherwise
            default : True

        replacement: dict
            nearest electrodes in dataset to replace missing ones
            in chs

        Returns
        -------
        no value, sets folds attribute with a list of arrays
        '''

        self.generate_set(replacement)

        self.n_subjects = len(self.target.epochs)

        if self._assert_partition(excl):
            raise Exception(
                f'Parition exceeds subjects count, use a different parition')

        n_phase = len(self.partition)

        if n_phase == 1:
            self.folds = [0]
        else:
            train_phase, val_phase = self.partition[0], self.partition[1]

            if n_phase == 2:
                # independent test set available
                test_phase = 0
            elif n_phase == 3:
                # generate a set set from the dataset
                test_phase = self.partition[2]
            else:
                # error : wrong partition
                raise AssertionError('Wrong partition scheme', self.partition)

            self.folds = self.get_folds(
                    nfolds, self.n_subjects, train_phase, val_phase,
                    test_phase, exclude_subj=excl)

    def run_evaluation(self, folds=None, pointer=None, check=False, savecsv=False, csvfolder=None):
        '''Perform evaluation on subsets of subjects.
        sets results

        Parameters
        ----------
        pointer : CheckPoint instance
            save state of evaluation

        check : bool
            if True, sets evaluation checkpoint for future operation resume,
            False otherwise

        Returns
        -------
        no value, sets results attribute
        '''
        # generate folds if folds are empty
        if not self.folds:
            self.generate_split(nfolds=1)

        if not pointer and check:
            pointer = CheckPoint(self)

        # res_acc, res_auc = [], []
        # res_tp, res_fp = [], []
        res = []

        if not self.model_compiled:
            self._compile_model()

        if self.log:
            print(f'Logging to file : {self.logger.handlers[0].baseFilename}')
            self.log_experiment()

        # if self.current:
        #     operations = range(self.current, len(self.folds))
        # else:
        #     operations = range(len(self.folds))

        operations = self.get_operations(folds)
        
        # for fold in range(len(self.folds)):
        for fold in operations:
            #
            if self.verbose == 0:
                print(f'Evaluating fold: {fold+1}/{len(self.folds)}...')

            rets = self._cross_set(fold)
            # if isinstance(rets, tuple):
            #     res_acc.append(rets[0])
            #     res_auc.append(rets[1])
            #     res_fp.append(rets[2])
            #     res_tp.append(rets[3])
            # else:
            #     res_acc.append(rets)

            # if self.log:
            #     msg = f' Fold : {fold+1} ACC: {res_acc[-1]}'
            #     if len(self.model.metrics) > 1:
            #         msg += f' AUC: {res_auc[-1]}'
            #     self.logger.debug(msg)
            #     self.logger.debug(
            #         f' Training stopped at epoch: {self.model_history.epoch[-1]}')            

            if self.log:
                msg = f" Subj : {fold+1} ACC: {rets['accuracy']}"
                # if len(self.model.metrics) > 1:
                if len(self.model_config['compile']['metrics']) > 1:
                    msg += f" AUC: {rets['auc']}"
                self.logger.debug(msg)
                self.logger.debug(f' Training stopped at epoch: {self.model_history.epoch[-1]}')

            res.append(rets)

            # if check:
            #     pointer.set_checkpoint(fold+1, self.model)

            if check:
                # pointer.set_checkpoint(fold+1, self.model)
                pointer.set_checkpoint(fold + 1, self.model, rets)

        if len(self.folds) == len(self.predictions):
            self.results = self.results_reports(res)
        elif check:
            self.results = self.results_reports(pointer.rets)

        if self.log:
            self._log_results()

        if savecsv:
            if self.results:
                self._savecsv(csvfolder)

        # Aggregate results
        # tfpr = {}
        # if res_auc:
            # res = np.array([res_acc, res_auc])
        #    res = {}
        #    res['acc'] = res_acc
        #    res['auc'] = res_auc
            # tfpr = np.array([res_fp, res_tp])
        #    tfpr['fp'] = res_fp
        #    tfpr['tp'] = res_tp
        # else:
        #    res = np.array(res_acc)
        #    tfpr = []

        # self.results = self.results_reports(res, tfpr)

    def get_folds(self, nfolds, population, tr, vl, ts, exclude_subj=True):
        """Generate train/validation/tes folds following Shuffle split strategy

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
        """

        folds = []

        # list : nfolds : [nsubjects_train] [nsubjects_val][nsubjects_test]
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
        #
        return folds

    def get_operations(self, folds=None):
        """get an iterable object for evaluation, it can be
        all folds or a defined subset of folds.
        In case of long evaluation, the iterable starts from the current
        index

        Parameters
        ----------
        folds : list | int, optional
            defined list of folds or a just a single one, by default None

        Returns
        -------
        range | list
            selection of folds to evaluate, from all folds available to a
            defined subset
        """
        if self.current and not folds:
            operations = range(self.current, len(self.folds))
        elif type(folds) is list:
            operations = folds
        elif type(folds) is int:
            operations = [folds]
        else:
            operations = range(len(self.folds))

        return operations

    def results_reports_old(self, res, tfpr={}):
        '''Collects evaluation results on a single dict

        Parameters
        ----------
        res : dict
            contains models performance
            - 'acc' : list
                Accuracy on all folds
            - 'auc' : list (only for binary class tasks)
                AUC on all folds

        tfpr : dict (only for binary class tasks)
            - 'fp' : False positives rate on all fold
            - 'tp' : True positives rate on all fold

        Returns
        -------
        results : dict of evaluation results compiled from models performance on the dataset
            - 'acc' : 2d array : Accuracy for each subjects on each folds (subjects x folds)
            - 'acc_mean' : double : Accuracy mean over all subjects and folds
            - 'acc_mean_per_fold' : 1d array : Accuracy mean per fold over all subjects
            For binary class tasks :
            - 'auc' : 2d array : AUC for each subjects on each folds (subjects x folds)
            - 'auc_mean' : double :  AUC mean over all subjects and folds
            - 'auc_mean_per_fold' :  1d array : AUC mean per fold over all subjects          
            - 'tpr' : 1d array : True posititves rate
            - 'fpr' : 1d array : False posititves rate
        '''
        if isinstance(self.target.epochs, np.ndarray):
            folds = len(self.folds)
            # subjects = self._get_n_subjects()
            # subjects = len(self.predictions)
            examples = len(self.predictions[0])
            dim = len(self.predictions[0][0])
            self.predictions = np.array(self.predictions).reshape((folds, examples, dim))
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
            results['fpr'] = tfpr['fp']
            results['tpr'] = tfpr['tp']
        else:
            # res : (subjects, folds)
            results['acc'] = res
            # mean across folds
            results['acc_mean_per_fold'] = res.mean(axis=0)
            # mean across subjects and folds
            results['acc_mean'] = res.mean()

        return results

    def _is_selectable(self, source=None):
        '''Find trials from source equivalent to target
        Parameters
        ----------
        source : dataset instance

        Returns
        -------
        ndarray of bool
            indices of samples to keep
        '''
        # y_target = np.unique(self.target.y[0])
        # labels = np.logical_and.reduce(np.unique(source.y[0]) == y_target)
        target_labels = self.target.labels_to_dict()
        source_labels = source.labels_to_dict()
        # return np.logical_and.reduce(np.unique(source.y[0]) == y_target)
        return target_labels == source

    def _cross_set(self, fold):
        '''Evaluate model on data split according to fold

        Parameters
        ----------
        fold : int
            fold index
        Returns
        -------
        rets : dict
            dict of evaluation results compiled from models performance on
            the dataset
            - 'acc' : 2d array : Accuracy for each subjects on each folds
            (subjects x folds)
            - 'acc_mean' : double : Accuracy mean over all subjects and folds
            - 'acc_mean_per_fold' : 1d array : Accuracy mean per fold over all
            subjects
            For binary class tasks :
            - 'auc' : 2d array : AUC for each subjects on each folds
            (subjects x folds)
            - 'auc_mean' : double :  AUC mean over all subjects and folds
            - 'auc_mean_per_fold' :  1d array : AUC mean per fold over all
             subjects
            - 'tpr' : 1d array : True posititves rate
            - 'fpr' : 1d array : False posititves rate
        '''
        split = self._split_set(fold)

        # normalize data
        # X_train, mu, sigma = fit_scale(split['X_train'])

        #if isinstance(split['X_val'], np.ndarray):
        #    X_val = transform_scale(split['X_val'], mu, sigma)
        # else:
        #    X_val = split['X_val']

        # X_test = transform_scale(split['X_test'], mu, sigma)
        # Y_train = split['Y_train']
        # Y_val = split['Y_val']
        # Y_test = split['Y_test']
        #
        # cws = class_weights(np.argmax(Y_train, axis=1))
        # evaluate model on subj on all folds
        # self.model_history, probs = self._eval_model(X_train, Y_train,
        #                                             X_val, Y_val, X_test,
        #                                             cws)
        # probs = self.model.predict(X_test)
        # rets = self.measure_performance(Y_test, probs)

        rets = self._eval_split(split)
        return rets

    def _diff(self, events):
        """Returns difference between events, used for SSVEP frequency stimulations

        Parameters
        ----------
        events : ndarray
            paradigm stimulus

        Returns
        -------
        int
            maxmimum difference between stimulus (in Hz if frequencies)
        """
        if isinstance(events, list):
            # some datasets attr are lists and not ndarray
            events = events[0]
        ev = np.unique(events)
        ev = np.array([float(ev[i]) for i in range(ev.size) if isfloat(ev[i])])
        d = np.unique(np.diff(sorted(ev)))
        if d.size > 1:
            return np.max(d)  # float conversion results in inconsistent values
        else:
            return d.item()

    def _split_set(self, fold):
        """fuse target and source data and return split of three sets for 
        Train/validation/test

        Parameters
        ----------
        fold : int
            fold index in the generated splits

        Returns
        -------
        split : dict
            phase data and their corresonding labels
            ndarrays :
            - X_train : train data
            - X_test  : test data
            - X_val   : validation data
            - Y_train : train labels
            - Y_test  : test labels
            - Y_val   : validation labels
            - classes : target class labels in numbers
        """
        # kernels = 1
        split = {}
        X_src = []
        Y_src = []

        # classes = np.unique(self.target.y)

        if hasattr(self.target, 'test_epochs'):
            if len(self.folds) == 1:

                X_t = self._flatten(self.target.epochs)
                Y_t = self._flatten(self.target.y)

                X_ts = self._flatten(self.target.test_epochs)
                Y_ts = self._flatten(self.target.test_y)

                X_v = None
                Y_v = None
            else:
                pass

        else:
            X_t = self._flatten(self.target.epochs[self.folds[fold][0]])
            # X_v = self._flatten(self.target.epochs[self.folds[fold][1]])
            X_ts = self._flatten(self.target.epochs[self.folds[fold][2]])
            Y_t = self._flatten(self.target.y[self.folds[fold][0]])
            # Y_v = self._flatten(self.target.y[self.folds[fold][1]])
            Y_ts = self._flatten(self.target.y[self.folds[fold][2]])
            if self._has_val():
                # X_val, Y_val = self._cat_lists(fold, 1)
                X_v = self._flatten(self.target.epochs[self.folds[fold][1]])
                X_v = X_v.transpose((2, 1, 0))
            else:
                X_v, Y_v = None, None
        #
        for src in self.source:
            X_src.append(self._flatten(src.epochs))
            Y_src.append(self._flatten(src.y))

        X_src = np.concatenate(X_src, axis=-1)
        Y_src = np.concatenate(Y_src, axis=-1)
        '''
        X_src = np.array(X_src).squeeze()        
        Y_src = np.array(Y_src).squeeze()
        '''
        X_t = np.concatenate((X_t, X_src), axis=-1)
        Y_t = np.concatenate((Y_t, Y_src), axis=-1)

        samples, channels, trials = X_t.shape

        # tr_s = X_ts.shape[-1]

        # X_t = X_t.transpose((2, 0, 1)).reshape((trials, kernels, channels, samples))
        # X_ts = X_ts.transpose((2, 0, 1)).reshape((tr_s, kernels, channels, samples))
        X_t = X_t.transpose((2, 1, 0))
        X_ts = X_ts.transpose((2, 1, 0))

        # if isinstance(X_v, np.ndarray):
            # tr_v = X_v.shape[-1]
            # X_v = X_v.transpose((2, 0, 1)).reshape((tr_v, kernels, channels, samples))
            # X_v = X_v.transpose((2, 1, 0))
            # Y_v = labels_to_categorical(Y_v)

        # FIXME : Training/Val/Test data has to be shuffled
        # np.random.shuffle (in-place shuffle)

        split['X_train'] = X_t
        split['X_test'] = X_ts
        split['X_val'] = X_v
        split['Y_train'] = Y_t
        split['Y_val'] = Y_v
        split['Y_test'] = Y_ts
        # split['Y_train'] = labels_to_categorical(Y_t)        
        # split['Y_test'] = labels_to_categorical(Y_ts)
        # split['classes'] = classes
        return split

    @staticmethod
    def _flatten(ndarray):
        """concatenate list of ndarrays with inconsistent number of elements in last dimension
        in a single ndarray

        Parameters
        ----------
        ndarray : list of ndarray
            arbitrary ndarray

        Returns
        -------
        ndarray :
            single ndarray with consistent number of elements in the last
            dimension
        """
        return np.concatenate([ndarray[idx] for idx in
                               range(len(ndarray))], axis=-1)

    def _get_min_channels(self):
        """find minimum channels shared between all datasets

        Returns
        -------
        list :
            minimum channels shared between all datasets
        """
        len_ch = [len(st.ch_names) for st in self.source]
        len_ch.append(len(self.target.ch_names))
        min_id = np.argmin(len_ch)

        if min_id.size > 1:
            min_id = min_id[0]

        if isinstance(min_id, np.ndarray):
            min_id = min_id.item()

        if min_id == len(len_ch) - 1:
            chs = self.target.ch_names
        else:
            chs = self.source[min_id].ch_names

        intersection = [list(set(src.ch_names).intersection(chs)) for src in self.source]
        intersection.append(list(set(self.target.ch_names).intersection(chs)))

        idx = np.argmin([len(ls) for ls in intersection])

        return intersection[idx]

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
        train_epochs = len(self.target.epochs)
        if hasattr(self.target, 'test_epochs'):
            test_epochs = len(self.target.test_epochs)
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
        if self.target:
            test_epochs = len(self.target.test_epochs) if hasattr(
                self.target, 'test_epochs') else 0
            if self._equal_subjects():
                return len(self.target.epochs)
            else:
                return len(self.target.epochs) + test_epochs
        else:
            return 0

    def _assert_partition(self, subjects=0, excl=False):
        """Assert if partition to be used do not surpass number of subjects available
        in dataset

        Parameters
        ----------
        subjects : int
            number of total subjects in target dataset

        excl : bool
            flag indicating whether the target subject is excluded from
            evaluation

        Returns
        -------
        bool : True if number of subjects is less than the sum of parition
            False otherwise
        """
        prt = np.sum(self.partition)
        return self.n_subjects < prt


