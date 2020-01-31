from aawedha.evaluation.base import Evaluation
from aawedha.evaluation.checkpoint import CheckPoint
from aawedha.analysis.utils import isfloat
import numpy as np


class CrossSet(Evaluation):
    '''
    '''
    # evl = SingleSubject(partition=[2,1], dataset=dt, verbose=0, lg=False)

    def __init__(self, source=[], target=[], partition=[],
                 verbose=2, lg=False):
        '''
        '''
        super().__init__(partition=partition,
                         verbose=verbose, lg=lg)
        self.source = source
        self.target = target

    def select_channels(self, ds=[], chs=[], replacement={}):
        '''
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

    def select_trials(self, source=[], ev=[]):
        '''
        '''
        ev = np.unique(self.target.events)

        d_source = self._diff(source.events)
        d_target = self._diff(ev)
        v = np.min([d_source, d_target])

        # events = np.unique(ev.astype(float))
        # events = np.unique(ev)
        # labels = np.unique(self.target.y)
        # new_labels = {str(events[i]): labels[i] for i in range(events.size)}
        # new_labels = {events[i]: labels[i] for i in range(events.size)}

        keys = self.target.events.flatten().tolist()
        values = self.target.y.flatten().tolist()
        new_labels = dict(zip(keys, values))

        source.rearrange(ev, v)
        source.update_labels(new_labels, v)

    def resample(self):
        '''
        '''
        fs_all = [src.fs for src in self.source]
        fs_all.append(self.target.fs)
        fs_all = np.array(fs_all)
        min_fs = fs_all.min()

        for src in self.source:
            src.resample(min_fs)

        self.target.resample(min_fs)

    def generate_set(self, replacement={}):
        '''
        '''
        #
        chs = self._get_min_channels()
        # ev = np.unique(self.target.events)

        # select channels and trials for source datasets
        for src in self.source:
            self.select_channels(src, chs)
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

        Returns
        -------
        no value, sets folds attribute with a list of arrays
        '''

        self.generate_set(replacement)

        self.n_subjects = len(self.target.epochs)

        if self._assert_partiton(excl):
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

    def run_evaluation(self, pointer=None, check=False):
        '''Perform evaluation on subsets of subjects

        Parameters
        ----------
        None

        Returns
        -------
        no value, sets results attribute
        '''
        # generate folds if folds are empty
        if not self.folds:
            self.generate_split(nfolds=1)

        if not pointer and check:
            pointer = CheckPoint(self)

        res_acc, res_auc = [], []
        res_tp, res_fp = [], []

        if not self.model_compiled:
            self._compile_model()

        if self.log:
            print(f'Logging to file : {self.logger.handlers[0].baseFilename}')
            self.log_experiment()

        if self.current:
            operations = range(self.current, len(self.folds))
        else:
            operations = range(len(self.folds))

        # for fold in range(len(self.folds)):
        for fold in operations:
            #
            if self.verbose == 0:
                print(f'Evaluating fold: {fold+1}/{len(self.folds)}...')

            rets = self._cross_set(fold)
            if isinstance(rets, tuple):
                res_acc.append(rets[0])
                res_auc.append(rets[1])
                res_fp.append(rets[2])
                res_tp.append(rets[3])
            else:
                res_acc.append(rets)

            if self.log:
                msg = f' Fold : {fold+1} ACC: {res_acc[-1]}'
                if len(self.model.metrics) > 1:
                    msg += f' AUC: {res_auc[-1]}'
                self.logger.debug(msg)
                self.logger.debug(
                    f' Training stopped at epoch: {self.model_history.epoch[-1]}')

            if check:
                pointer.set_checkpoint(fold+1, self.model)

        # Aggregate results
        if res_auc:
            res = np.array([res_acc, res_auc])
            tfpr = np.array([res_fp, res_tp])
        else:
            res = np.array(res_acc)
            tfpr = []

        self.results = self.results_reports(res, tfpr)

    def results_reports(self, res, tfpr={}):
        '''
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

    def _cross_set(self, fold):
        '''
        '''
        split = self._split_set(fold)
        # normalize data
        X_train, mu, sigma = self.fit_scale(split['X_train'])

        if isinstance(split['X_val'], np.ndarray):
            X_val = self.transform_scale(split['X_val'], mu, sigma)
        else:
            X_val = split['X_val']

        X_test = self.transform_scale(split['X_test'], mu, sigma)
        Y_train = split['Y_train']
        Y_val = split['Y_val']
        Y_test = split['Y_test']
        #
        cws = self.class_weights(np.argmax(Y_train, axis=1))
        # evaluate model on subj on all folds
        self.model_history, probs = self._eval_model(X_train, Y_train,
                                                     X_val, Y_val, X_test,
                                                     cws)
        # probs = self.model.predict(X_test)
        rets = self.measure_performance(Y_test, probs)

        return rets

    def _diff(self, events):
        '''
        '''
        if isinstance(events, list):
            # some datasets attr are lists and not ndarray
            events = events[0]
        ev = np.unique(events)
        ev = np.array([float(ev[i]) for i in range(ev.size) if isfloat(ev[i])])
        d = np.unique(np.diff(sorted(ev)))
        if d.size > 1:
            return np.max(d)  # float converion results in inconsisten values
        else:
            return d.item()

    def _split_set(self, fold):
        '''
        '''
        kernels = 1
        split = {}
        X_src = []
        Y_src = []

        classes = np.unique(self.target.y)

        if hasattr(self.target, 'test_epochs'):
            if len(self.folds) == 1:

                X_t = self._flatten(self.target.epochs)
                Y_t = self._flatten(self.target.y)

                X_ts = self._flatten(self.target.test_epochs)
                Y_ts = self._flatten(self.target.test_y)

                X_v = None
                Y_v = None

        else:
            X_t = self._flatten(self.target.epochs[self.folds[fold][0]])
            X_v = self._flatten(self.target.epochs[self.folds[fold][1]])
            X_ts = self._flatten(self.target.epochs[self.folds[fold][2]])

            Y_t = self._flatten(self.target.y[self.folds[fold][0]])
            Y_v = self._flatten(self.target.y[self.folds[fold][1]])
            Y_ts = self._flatten(self.target.y[self.folds[fold][2]])

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

        tr_s = X_ts.shape[-1]

        X_t = X_t.transpose((2, 0, 1)).reshape((trials, kernels, channels, samples))
        X_ts = X_ts.transpose((2, 0, 1)).reshape((tr_s, kernels, channels, samples))

        if isinstance(X_v, np.ndarray):
            tr_v = X_v.shape[-1]
            X_v = X_v.transpose((2, 0, 1)).reshape((tr_v, kernels, channels, samples))            
            Y_v = self.labels_to_categorical(Y_v)

        split['X_train'] = X_t
        split['X_test'] = X_ts
        split['X_val'] = X_v
        split['Y_val'] = Y_v
        split['Y_train'] = self.labels_to_categorical(Y_t)        
        split['Y_test'] = self.labels_to_categorical(Y_ts)
        split['classes'] = classes

        return split

    def _flatten(self, ndarray):
        '''
        '''
        return np.concatenate([ndarray[idx] for idx in
                               range(len(ndarray))], axis=-1)

    def _get_min_channels(self):
        '''
        '''
        len_ch = [len(st.ch_names) for st in self.source]
        len_ch.append(len(self.target.ch_names))
        len_ch = np.array(len_ch)
        min_id = np.where(len_ch == len_ch.min())[0]

        if min_id.size > 1:
            min_id = min_id[0]

        if isinstance(min_id, np.ndarray):
            min_id = min_id.item()

        if min_id == len(len_ch) - 1:
            return self.target.ch_names
        else:
            return self.source[min_id].ch_names

    def _assert_partiton(self, subjects=0, excl=False):
        '''Assert if partition to be used do not surpass number of subjects available
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
        '''
        prt = np.sum(self.partition)
        return self.n_subjects < prt
