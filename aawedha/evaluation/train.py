from aawedha.evaluation.evaluation_utils import create_split
from sklearn.model_selection import train_test_split
from aawedha.evaluation.base import Evaluation
from aawedha.utils.utils import time_now
import pandas as pd
import numpy as np
import os


class Train(Evaluation):

    def run_evaluation(self, val_size=0.1, selection=None, save_model=False, 
                       model_format='TF', save_history=False, folder=None,
                       new_labels=None, events=None):
        """Performs training

        Parameters
        ----------
        val_size : float, optional
            portion of dataset to be included in the validation split, by default 0.1.
            should be between 0.0 and 1.0
        
        selection : list, optional
            subjects data to be kept for training, by default None
        
        save_model : bool, optional
            if True, save model after training as a the model_format specified, by default False

        model_format : str, optional
            model saving format either TF SavedModel of Keras HDF5, by default TF SavedModel.
                
        save_history : bool, optional
            if True, save model's training history as a pandas DataFrame. by default False
        
        folder : str, optional
            folder where to save both the model and its history, by default None

        new_labels : dict, optional
            dictionary of new labels for the dataset based on the events to keep, by default None
            key: str
                events e.g. 10, 12, frequency stimulations in SSVEP paradigm
            values: float
                classe labels 
        
        events : array of str, optional
            stimulation events to keep, by default None
        """
        if not self.model_compiled:
            self._compile_model()

        if self.verbose == 0:
              print(f"Training on DataSet: {self.dataset.title}...")

        # select a subset of trials
        if new_labels:
            self._select_trials(new_labels, events)

        self.results = self._train(selection, val_size)

        # save model in HDF5 format or SavedModel
        if save_model:
          self.save_model(folderpath=folder, modelformat=model_format)

        # save training history as a DataFrame
        if save_history:
          self._save_history(folder=folder)

    def _select_trials(self, new_labels=None, events=None):
        """Keep a subset of trials for training

        Parameters
        ----------
        new_labels : dict, optional
            dictionary of new labels for the dataset based on the events to keep, by default None
            key: str
                events e.g. 10, 12, frequency stimulations in SSVEP paradigm
            values: float
                classe labels 
        
        events : array of str, optional
            stimulation events to keep, by default None
        """
        if new_labels:
            self.dataset.rearrange(events)
            self.dataset.update_labels(new_labels) 

    def _train(self, selection, val_size):
        """Split data and train model

        Parameters
        ----------
         selection : list, optional
            subjects data to be kept for training, by default None

        val_size : float, optional
            portion of dataset to be included in the validation split, by default 0.1.
            should be between 0.0 and 1.0

        Returns
        -------
        list
            performance results
        """
        split = self._split_data(selection, val_size)
        rets = self._eval_split(split)
        return rets

    def _split_data(self, selection=None, val_size=0.1):
        """Splits subsets of Subjects data to be evaluated into
        train/validation/test sets following the indices specified in the fold

        Parameters
        ----------
        selection : list, optional
            subjects data to be kept for training, by default None

        val_size : float, optional
            portion of dataset to be included in the validation split, by default 0.1.
            should be between 0.0 and 1.0

        Returns
        -------
        split : dict of nd arrays
            X_train, Y_train, X_Val, Y_Val, X_test, Y_test
            train/validation/test EEG data and labels
            classes : array of values used to denote class labels
        """        
        if selection:
          if isinstance(self.dataset.epochs, np.ndarray):
            X = self.dataset.epochs[selection]
            Y = self.dataset.y[selection]
          else:
            X = [self.dataset.epochs[i] for i in selection]
            Y = [self.dataset.y[i] for i in selection]
        else:
          X = self.dataset.epochs
          Y = self.dataset.y

        shape = (2, 1, 0)
        X, Y = self._cat_lists(X, Y)
        X = X.transpose(shape)
        if val_size == 0:
            X_train, X_val, Y_train, Y_val = X, None, Y, None
        else:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size)

        if Y_train.min() != 0:
            Y_train -= 1
            if isinstance(Y_val, np.ndarray):
                Y_val -= 1

        split = create_split(X_train, X_val, None, Y_train, Y_val, None)
        return split

    def _cat_lists(self, X, Y):
        """Concatenate lists into a single Tensor

        Parameters
        ----------
        X : ndarray (subjects, samples, channels, trials)
            EEG data 
        Y : ndarray (subject, trials)
            class labels

        Returns
        -------
        X : ndarray (samples, channels, subjects*trials)
            EEG data concatenated
        Y : ndarray (subject*trials)
            class labels
        """
        if isinstance(X, list):
            n_selection = len(X)
        else:
            n_selection = X.shape[0]
        X = np.concatenate([X[idx] for idx in range(n_selection)], axis=-1)
        Y = np.concatenate([Y[idx] for idx in range(n_selection)], axis=-1)
        return X, Y

    def _save_history(self, folder=None):
        """Save model training history

        Parameters
        ----------
        folder : str
            saving folder path, default: trained/history
        """
        date = time_now()
        if not folder:
            if not os.path.isdir('trained/history'):
                os.mkdir('trained/history')
            folder = 'trained/history'
        fname = os.path.join(folder, '_'.join([self.model.name, date]))
        df = pd.DataFrame(self.model_history.history)
        df.to_csv(fname, encoding='utf-8')
        