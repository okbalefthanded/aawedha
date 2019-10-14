from aawedha.evaluation.base import Evaluation
from tensorflow.keras import utils as np_utils
from sklearn.metrics import roc_auc_score
import numpy as np
import random 


class SingleSubject(Evaluation):

    
    def generate_split(self, nfolds=30):
        '''
        '''
        # folds = []
        n_phase = len(self.partition)
        train_phase, val_phase = self.partition[0], self.partition[1]
  
        if type(self.dataset.y) is list:
            # 
            pass
        else:
            n_trials = self.dataset.y.shape[1]

        if n_phase == 2:
            # independent test set available    
            test_phase = 0
        elif n_phase == 3:
            # generate a set set from the dataset
            test_phase = self.partition[2]    
        else:
            # error : wrong partition
            raise AssertionError('Wrong partition scheme', self.partition)
    
        part = np.round(n_trials / np.sum(self.partition)).astype(int)
  
        train_phase = train_phase * part
        val_phase = val_phase * part
        test_phase = test_phase * part
  
        # for _ in range(nfolds):
        #    tmp = np.array(random.sample(range(n_trials), n_trials))
        #    folds.append([tmp[:train_phase], tmp[train_phase:train_phase+val_phase], tmp[-test_phase:]])

        self.folds = self.get_folds(nfolds, n_trials, train_phase, val_phase, test_phase)

    def run_evaluation(self):
        '''
        '''        
        # generate folds if folds are empty
        if not self.folds:
            self.folds = self.generate_split(nfolds=30)
        # 
        res = []
        for subj in range(self.n_subjects):
            res_per_subject, avg_res = self._single_subject(subj)
            res.append(res_per_subject)        
        # Aggregate results
        res = np.array(res)
        
        self.results = self.results_reports(res)
        # self.results['results_acc'] = res[:,0]
        # self.results['results_auc'] = res[:,1]    

    def _single_subject(self, subj):
        '''
        '''
        # prepare data
        x = self.dataset.epochs[subj, :, :, :]
        y = self.dataset.y[subj, :]
        samples, channels, trials = x.shape
        x = x.transpose((2,1,0))
        kernels = 1 # 
        x = x.reshape((trials, kernels, channels, samples))
        y = np_utils.to_categorical(y)  
        res = []             
        # get in the fold!!!
        for fold in range(len(self.folds)):
            X_train = x[self.folds[fold][0],:,:,:]
            X_val   = x[self.folds[fold][1],:,:,:]
            X_test  = x[self.folds[fold][2],:,:,:]
            
            Y_train = y[self.folds[fold][0]]
            Y_val   = y[self.folds[fold][1]]
            Y_test  = y[self.folds[fold][2]]
            # normalize data
            X_train, mu, sigma = self.fit_scale(X_train)
            X_val = self.transform_scale(X_val, mu, sigma)
            X_test = self.transform_scale(X_test, mu, sigma)
            #
            class_weights = self.class_weights(np.argmax(Y_train, axis=1))
            # evaluate model on subj on all folds
            # rename the fit method
            self.model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
                            verbose = 2, validation_data=(X_val, Y_val),
                            class_weight = class_weights)
            # train/val                
            # test 
            probs = self.model.predict(X_test)
            preds = probs.argmax(axis = -1)  
            acc   = np.mean(preds == Y_test.argmax(axis=-1))
            auc_score = roc_auc_score(Y_test.argmax(axis=-1), preds)
            res.append([acc, auc_score])    
        
        # res = []  # shape: (n_folds, 2)           
        # average performance on all folds
        res = np.array(res)     
        return res, res.mean(axis=0)    