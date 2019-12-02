'''
Base class for evaluations

'''
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import random
import abc
import os


class Evaluation(object):

    def __init__(self, dataset=None, partition=[], folds=[],
                 model=None, verbose=2):
        '''
        '''
        self.dataset = dataset
        self.partition = partition
        self.folds = folds
        self.model = model
        self.predictions = []
        self.cm = []  # confusion matrix per fold
        self.results = {}  # dict
        self.model_history = {}
        self.verbose = verbose
        self.n_subjects = self._get_n_subjects()

    @abc.abstractmethod
    def generate_split(self, nfolds):
        '''
        '''
        pass

    @abc.abstractmethod
    def run_evaluation(self):
        '''
        '''
        pass

    def measure_performance(self, Y_test, probs):
        '''
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
        '''
        '''
        folds = len(self.folds)
        # subjects = self._get_n_subjects()
        # subjects = len(self.predictions)
        examples = len(self.predictions[0])
        dim = len(self.predictions[0][0])
        self.predictions = np.array(self.predictions).reshape(
                            (self.n_subjects, folds, examples, dim))
        #
        results = {}
        #
        if tfpr:
            # res : (metric, subjects, folds)
            # means = res.mean(axis=-1) # mean across folds
            r1 = np.array(res['acc'])
            results['acc'] = r1
            results['acc_mean_per_fold'] = r1.mean(axis=0)
            results['acc_mean_per_subj'] = r1.mean(axis=1)
            results['acc_mean'] = r1.mean()
            #
            r2 = np.array(res['auc'])
            results['auc'] = r2
            results['auc_mean_per_fold'] = r2.mean(axis=0)
            results['auc_mean_per_subj'] = r2.mean(axis=1)
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
            results['acc_mean_per_subject'] = res.mean(axis=1)
            results['acc_mean'] = res.mean()

        return results

    def get_folds(self, nfolds, population, tr, vl, ts, exclude_subj=True):
        '''
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
                                      np.array(selection[tr:])])
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
            for _ in range(nfolds):
                tmp = np.array(random.sample(range(population), population))
                folds.append([tmp[:tr], tmp[tr:tr + vl], tmp[-ts:]])
        #
        return folds

    def fit_scale(self, X):
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        X = np.subtract(X, mu[None, :, :])
        X = np.divide(X, sigma[None, :, :])
        return X, mu, sigma

    def transform_scale(self, X, mu, sigma):
        X = np.subtract(X, mu[None, :, :])
        X = np.divide(X, sigma[None, :, :])
        return X

    def class_weights(self, y):
        '''
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
        '''
        '''
        classes = np.unique(y)
        if np.isin(0, classes):
            y = to_categorical(y)
        else:
            y = to_categorical(y - 1)
        return y

    def save_model(self, folderpath=None):
        '''
        '''
        if not os.path.isdir('trained'):
            os.mkdir('trained')
        if not folderpath:
            folderpath = 'trained'
        prdg = self.dataset.paradigm.title
        dt = self.dataset.title
        filepath = folderpath + '/' + '_'.join(['model', prdg, dt, '.h5'])
        self.model.save(filepath)

    def _equale_subjects(self):
        '''
        '''
        ts = 0
        tr = len(self.dataset.epochs)
        if hasattr(self.dataset, 'test_epochs'):
            ts = len(self.dataset.test_epochs)
        return tr == ts

    def _get_n_subjects(self):
        '''
        '''
        ts = len(self.dataset.test_epochs) if hasattr(self.dataset, 'test_epochs') else 0
        if self._equale_subjects():
            return len(self.dataset.epochs)
        else:
            return len(self.dataset.epochs) + ts
