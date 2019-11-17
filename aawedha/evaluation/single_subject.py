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
            n_trials = np.array([self.dataset.y[i].shape[0] for i in range(self.n_subjects)])
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
        # equal_subjects = self._get_n_subj()

        if hasattr(self.dataset, 'test_epochs'):

            if self._equale_subjects():
                independent_test = True
            else:
                # concatenate train & test data
                # test data are different subjects 
                n_subj  = self._fuse_data()
                self.n_subjects = n_subj

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
        x = self.dataset.epochs[subj][:, :, :]
        y = self.dataset.y[subj][:]
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
            #
            split = self._split_set(x, y, subj, fold, indie)
            # normalize data
            X_train, mu, sigma = self.fit_scale(split['X_train'])
            X_val = self.transform_scale(split['X_val'], mu, sigma)
            X_test = self.transform_scale(split['X_test'], mu, sigma)
            #
            Y_train = split['Y_train']
            Y_test  = split['Y_test']
            Y_val   = split['Y_val']
            #
            class_weights = self.class_weights(np.argmax(Y_train, axis=1))
            # evaluate model on subj on all folds
            # rename the fit method
            self.model_history = self.model.fit(X_train, Y_train, batch_size = 64, epochs = 500, 
                            verbose = self.verbose, validation_data=(X_val, Y_val),
                            class_weight = class_weights)
            # train/val                
            # test 
            probs = self.model.predict(X_test)
            preds = probs.argmax(axis = -1)  
            acc   = np.mean(preds == Y_test.argmax(axis=-1))
            res_acc.append(acc.item())
            if classes.size == 2:
                auc_score = roc_auc_score(Y_test.argmax(axis=-1), preds)
                res_auc.append(auc_score.item())                
        
        # res = []  # shape: (n_folds, 2)        
        return res_acc, res_auc 

    def _fuse_data(self):
        '''
        '''
        # TODO : when epochs/y/test_epochs/y_test are lists???
        # make lists of lists
        if type(self.dataset.epochs) is list:
            # TODO
            pass
        else:
            self.dataset.epochs = np.vstack((self.dataset.epochs, self.dataset.test_epochs))
            self.dataset.y = np.vstack((self.dataset.y, self.dataset.test_y))
        return self.dataset.epochs.shape[0] # n_subject 

    def _equale_subjects(self):
        '''
        '''
        ts = 0
        tr = len(self.dataset.epochs)
        if hasattr(self.dataset, 'test_epochs'):          
            ts = len(self.dataset.test_epochs)                   
        return tr == ts

    def _split_set(self, x=None, y=None, subj=0, fold=0, indie=False):
        '''
        '''
        # folds[0][0][0] : inconsistent fold subject trials
        # folds[0][0] : same trials numbers for all subjects
        split = {}
        trials, kernels, channels, samples = x.shape
        
        if type(self.dataset.epochs) is list:
            f = self.folds[fold][subj][:]
        else:
            f = self.folds[fold][:]
            
        X_train = x[f[0],:,:,:]
        X_val   = x[f[1],:,:,:]
        Y_train = y[f[0]]
        Y_val   = y[f[1]]
        if indie:
            trs = self.dataset.test_epochs[0].shape[2]
            X_test = self.dataset.test_epochs[subj][:,:,:].transpose((2,1,0))
            X_test  = X_test.reshape((trs, kernels, channels, samples))
            Y_test = self.labels_to_categorical(self.dataset.test_y[subj][:])
        else:
            X_test  = x[f[2],:,:,:]
            Y_test  = y[f[2]]

        split['X_train'] = X_train
        split['X_val']   = X_val
        split['X_test']  = X_test
        split['Y_train'] = Y_train
        split['Y_val']   = Y_val
        split['Y_test']  = Y_test
        
        return split
        