'''
Base class for evaluations

'''
from tensorflow.keras.utils import to_categorical 
import numpy as np
import random
import abc

class Evaluation(object):

    def __init__(self, n_subjects=0, partition=[], 
                folds=[], dataset=None, 
                model=None, verbose=2
                ):
        
        self.n_subjects = n_subjects
        self.partition = partition
        self.folds = folds
        self.dataset = dataset
        self.model = model
        self.results = {} # dict 
        self.verbose = verbose
        

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

    def results_reports(self, res):
        '''
        '''
        results = {}
        print(f'raw results : {res}')
        mean_res = res.mean(axis=0)
        if mean_res.ndim == 2:
            results['acc'] = mean_res[:,0]
        else:
            results['acc'] = mean_res[0]

        if res.size == 2: 
            # binary classification
            results['auc'] = mean_res[:,1]
        
        print(f'results : {results}')
        return results   

    def get_folds(self, nfolds, population, tr, vl, ts, exclude_subj=True):
        '''
        '''

        folds = []
        if hasattr(self.dataset, 'test_epochs'):
            if exclude_subj:
                # fully cross-subject, no subject train data in fold
                # nfolds x nsubjects
                pass
            else:
                # nfolds
                pass
        else:
            for _ in range(nfolds):
                tmp = np.array(random.sample(range(population), population))
                folds.append([tmp[:tr], tmp[tr:tr+vl], tmp[-ts:]])
        
        return folds

    def fit_scale(self, X):
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        X = np.subtract(X, mu[None,:,:])
        X = np.divide(X, sigma[None,:,:])
        return X, mu, sigma
        

    def transform_scale(self, X, mu, sigma):
        X = np.subtract(X, mu[None,:,:])
        X = np.divide(X, sigma[None,:,:])
        return X

    def class_weights(self, y):
        '''
        '''
        cl_weights = {}
        classes = np.unique(y)
        n_perclass = [np.sum(y==cl) for cl in classes]
        n_samples = np.sum(n_perclass)
        ws = np.array([np.ceil(n_samples / cl).astype(int) for cl in n_perclass])
        if np.unique(ws).size == 1:
            # balanced classes
            cl_weights = {cl:1 for cl in classes}
        else:
            # unbalanced classes
            if classes.size == 2:
                cl_weights = {classes[ws == ws.max()].item():ws.max(), classes[ws < ws.max()].item():1}
            else:
                cl_weights = {cl:ws[idx] for idx,cl in enumerate(classes)}
        return cl_weights

    def labels_to_categorical(self, y):
        '''
        '''
        classes = np.unique(y)
        if np.isin(0, classes):
            y = to_categorical(y)
        else:
            y = to_categorical(y-1)
        return y

