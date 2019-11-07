from aawedha.evaluation.base import Evaluation
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
  
        self.folds = self.get_folds(nfolds, n_trials, train_phase, val_phase, test_phase, exclude_subj=False)

    def run_evaluation(self, subject=None):
        '''
        '''        
        # generate folds if folds are empty
        if not self.folds:
            self.folds = self.generate_split(nfolds=30)
        # 
        res_acc = []
        res_auc = []

        independent_test = False
        train_subjects = self.dataset.epochs.shape[0]

        if hasattr(self.dataset, 'test_epochs'):
            if train_subjects == self.dataset.test_epochs.shape[0]:
                independent_test = True

        if subject:
            operations = [subject]
        else:    
            operations = range(self.n_subjects)

        for subj in operations:
            # res_per_subject, avg_res = self._single_subject(subj, independent_test)
            acc_per_subject, auc_per_subject = self._single_subject(subj, independent_test)
            res_acc.append(acc_per_subject)
            if auc_per_subject:
                res_auc.append(auc_per_subject)        
        
        # Aggregate results
        # res_acc = np.array(res_acc)
        if res_auc:
            res = np.array([res_acc, res_auc])
        else:
            res = np.array(res_acc)
            
        self.results = self.results_reports(res)
 

    def _single_subject(self, subj, indie=False):
        '''
        '''
        # prepare data
        x = self.dataset.epochs[subj, :, :, :]
        y = self.dataset.y[subj, :]
        samples, channels, trials = x.shape
        x = x.transpose((2,1,0))
        kernels = 1 # 
        x = x.reshape((trials, kernels, channels, samples))
        # 
        classes = np.unique(y)
        y = self.labels_to_categorical(y)
        res_acc = []               
        res_auc = []
        # get in the fold!!!
        for fold in range(len(self.folds)):
            X_train = x[self.folds[fold][0],:,:,:]
            X_val   = x[self.folds[fold][1],:,:,:]
            Y_train = y[self.folds[fold][0]]
            Y_val   = y[self.folds[fold][1]]

            # if hasattr(self.dataset, 'test_epochs'):
            if indie:
                trs = self.dataset.test_epochs.shape[3]
                X_test = self.dataset.test_epochs[subj,:,:,:].transpose((2,1,0))
                X_test  = X_test.reshape((trs, kernels, channels, samples))
                Y_test = self.labels_to_categorical(self.dataset.test_y)
            else:
                X_test  = x[self.folds[fold][2],:,:,:]
                Y_test  = y[self.folds[fold][2]]          
            
            # normalize data
            X_train, mu, sigma = self.fit_scale(X_train)
            X_val = self.transform_scale(X_val, mu, sigma)
            X_test = self.transform_scale(X_test, mu, sigma)
            #
            class_weights = self.class_weights(np.argmax(Y_train, axis=1))
            # evaluate model on subj on all folds
            # rename the fit method
            self.model.fit(X_train, Y_train, batch_size = 64, epochs = 500, 
                            verbose = self.verbose, validation_data=(X_val, Y_val),
                            class_weight = class_weights)
            # train/val                
            # test 
            probs = self.model.predict(X_test)
            preds = probs.argmax(axis = -1)  
            acc   = np.mean(preds == Y_test.argmax(axis=-1))
            res_acc.append(acc)
            if classes.size == 2:
                auc_score = roc_auc_score(Y_test.argmax(axis=-1), preds)
                res_auc.append(auc_score)                
        
        # res = []  # shape: (n_folds, 2)           
        # average performance on all folds
        # res_acc = np.array(res_acc)             
        return res_acc, res_auc # res.mean(axis=0)
            