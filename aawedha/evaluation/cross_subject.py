from aawedha.evaluation.base import Evaluation
from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow.keras.utils as np_utils
import random

class CrossSubject(Evaluation):


    def generate_split(self, nfolds=30):
        '''
        '''
        # folds = []
        n_phase = len(self.partition)
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

        self.folds = self.get_folds(nfolds, self.n_subjects, train_phase, val_phase, test_phase)


    def run_evaluation(self):
        '''
        '''
        # generate folds if folds are empty
        if not self.folds:
            self.folds = self.generate_split(nfolds=30)
        # 
        res = []
        for fold in range(self.folds):
            res_per_fold = self._cross_subject(fold)
            res.append(res_per_fold)        
        
        # Aggregate results
        res = np.array(res)      
        self.results = self.results_reports(res)  

    def _cross_subject(self, fold):
        '''
        '''
        kernels = 1
        x = self.dataset.epochs
        subjects, samples, channels, trials = x.shape  
        y = self.dataset.y
        x = x.transpose((0,3,2,1))
        y = np_utils.to_categorical(y)
        #
        if  len(self.partition) == 3:      
            tr, val, ts = self.partition
        else :
            # TODO
            pass  
        #        
        X_train = x[self.folds[fold][0],:,:,:].reshape((tr*trials,kernels,channels,samples))
        X_val   = x[self.folds[fold][1],:,:,:].reshape((val*trials,kernels,channels,samples))
        X_test  = x[self.folds[fold][2],:,:,:].reshape((ts*trials,kernels,channels,samples))
    
        Y_train = y[self.folds[fold][0],:].reshape((tr*trials,2))
        Y_val   = y[self.folds[fold][1],:].reshape((val*trials,2))
        Y_test  = y[self.folds[fold][2],:].reshape((ts*trials,2))
    
        # normalize data
        X_train, mu, sigma = self.fit_scale(X_train)
        X_val = self.transform_scale(X_val, mu, sigma)
        X_test = self.transform_scale(X_test, mu, sigma)
        #
        cws = self.class_weights(np.argmax(Y_train, axis=1))
        # evaluate model on subj on all folds
    
        self.model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
              verbose = 0, validation_data=(X_val, Y_val),
              class_weight = cws)
            # train/val
        probs = self.model.predict(X_test)
        preds = probs.argmax(axis = -1)  
        acc   = np.mean(preds == Y_test.argmax(axis=-1))
        auc_score = roc_auc_score(Y_test.argmax(axis=-1), preds)    
        return np.array([acc, auc_score])
        
