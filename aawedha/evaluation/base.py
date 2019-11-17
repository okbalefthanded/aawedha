'''
Base class for evaluations

'''
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
import abc
import os

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
                
        if res.ndim == 3:
            # res : (metric, subjects, folds)
            means = res.mean(axis=-1) # mean across folds
            results['acc'] = means[0]
            results['acc_mean'] = means[0].mean()
            results['auc'] = means[1]
            results['auc_mean'] = means[1].mean()
        else:
            # res : (subjects, folds)
            results['acc'] = res.mean(axis=-1) # mean across folds
            results['acc_mean'] = res.mean() # mean across subjects

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
                        folds.append([np.array(selection[:tr]), np.array(selection[tr:])])            
            elif self.__class__.__name__ == 'SingleSubject':
                # generate folds for test set from one set                
                pop = population
                t = tr
                v = vl
                s = ts                    
                # generate folds for test set from one set
                for f in range(nfolds):
                    if type(population) is np.ndarray:
                        # inconsistent numbers of trials among subjects          
                        sbj = []
                        for subj in range(self.n_subjects):            
                            pop = population[subj]
                            t = tr[subj]
                            v = vl[subj]
                            s = ts[subj]
                            tmp = np.array(random.sample(range(pop), pop))            
                            sbj.append([tmp[:t], tmp[t:t+v], tmp[-s:]]) 
                        folds.append(sbj)
                    else: 
                        # same numbers of trials for all subjects         
                        tmp = np.array(random.sample(range(pop), pop))
                        folds.append([tmp[:t], tmp[t:t+v], tmp[-s:]])            
                              
                #for _ in range(nfolds):
                #    tmp = np.array(random.sample(range(population), population))
                #    folds.append([tmp[:tr], tmp[tr:tr+vl], tmp[-ts:]])                         
        else:
            # generate folds for test set from one set
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

    def save_model(self, folderpath=None):
        '''
        '''
        if not os.path.isdir('trained'):
            os.mkdir('trained')
        if not folderpath:            
           folderpath = 'trained' 
        prdg = self.dataset.paradigm.title
        dt = self.dataset.title
        filepath = folderpath + '/' + '_'.join(['model',prdg,dt,'.h5'])    
        self.model.save(filepath)
        
