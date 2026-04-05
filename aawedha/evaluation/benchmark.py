from aawedha.paradigms.utils_paradigms import phrase_from_dataset
from aawedha.evaluation.evaluation_utils import aggregate_results
from aawedha.evaluation.checkpoint import CheckPoint
from aawedha.evaluation.base import Evaluation
from aawedha.io.base import DataSet
import numpy as np
import torch
import abc


class BenchMark(Evaluation):
    
    #pylint: disable=too-many-arguments
    def run_evaluation(self, 
                       selection=None, 
                       pointer=None, 
                       check=False,
                       savecsv=False, 
                       csvfolder=None,
                       dfname=None):
        """Perform evaluation on each subject

        Parameters
        ----------
        selection : int | list
            - specific subject id, performs a single evaluation.
            - list of subjects from the set of subjects available in dataset
            default : None, evaluate each subject

        pointer : CheckPoint instance
            saves the state of evaluation

        check : bool
            if True, sets evaluation checkpoint for future operation resume,
            False otherwise

        savecsv: bool
            if True, saves evaluation results in a csv file as a pandas DataFrame

        csvfolder : str
            if savecsv is True, the results files in csv will be saved inside this folder
        """
        operations, pointer = self._pre_operations(selection, pointer, check)
        eval_results = self.execute(operations, check, pointer)
        
        if (not isinstance(self.dataset.epochs, list) and
                self.dataset.epochs.ndim == 3):
            self.dataset.recover_dim()

        if isinstance(self.dataset, DataSet):
            classes = self.dataset.get_n_classes()
        else:
            # for experimental CrossSet evaluation
            classes = self.target.get_n_classes()

        self.score.results_reports(eval_results, classes, {self._eval_type(): list(operations)})        
        self._post_operations(savecsv, csvfolder, dfname)

        return self

    def execute(self, operations, check, pointer):
        """Execute the evaluations on specified folds in operations.

        Parameters
        ----------
        operations : Iterable
            range | list, specify index of folds to evaluate.        
       
        check : bool
            if True, sets evaluation checkpoint for future operation resume,
            False otherwise.
        
        pointer : CheckPoint instance
            saves the state of evaluation

        Returns
        -------
        list
            list of each fold performance following the metrics specified in the model config.
        """
        eval_results = []
        for op in operations:
            
            if self.settings.verbose == 0:
                print(f'Evaluating {self._eval_type()}: {op+1}/{self._total_operations()}...')

            op_results = self._eval_operation(op)

            if self.log:
                self._log_operation_results(op, op_results)

            eval_results.append(op_results)

            if check:
                pointer.set_checkpoint(op + 1, self.learner.model, op_results)
        
        return eval_results

    def get_operations(self, selection=None):
        """get an iterable object for evaluation, it can be
        all folds or a defined subset of folds.
        In case of long evaluation, the iterable starts from the current
        index

        Parameters
        ----------
        selection : list | int, optional
            defined list of folds or a just a single one, by default None

        Returns
        -------
        Iterator : range | list
            selection of folds to evaluate, from all folds available to a
            defined subset
        """
        total_operations = self._total_operations()
        if self.settings.current and not selection:
            operations = range(self.settings.current, total_operations)
        elif isinstance(selection, list):
            operations = selection
        elif isinstance(selection, int):
            operations = [selection]
        else:
            operations = range(total_operations)

        return operations 
    
    def _pre_operations(self, selection=None, pointer=None, check=False):
        """Preparations before evaluation:
        - generate splits if empty.
        - set checkpoint if check is required.
        - compile model if newly created or reset.
        - log evaluation info.

        Parameters
        ----------
        selection : int or list, optional
            folds/subject selections to be evaluated, by default None
            if None all will be evaluated.
        pointer : Checkpoint instance, optional
            saving checkpoint, by default None
        check : bool, optional
            if True , and pointer is None. Create a Checkpoint, by default False.

        Returns
        -------
        operations: Iterator
            indexes of Folds/subjects to evaluate.
        pointer : Checkpoint instance
            saving checkpoint
        """
        # generate folds if folds are empty
        if not self.settings.folds and self.settings.partition:
            self.generate_split(nfolds=30)

        if not pointer and check:
            pointer = CheckPoint(self)

        operations = self.get_operations(selection)

        if not self.learner.compiled:
            self._compile_model()

        if self.log:
            print(f'Logging to file : {self.logger.name()}')
            self.log_experiment()
        
        return operations, pointer
    
    def _eval_paradigm_metrics(self, perf, op):
        """Evaluate paradigm specific metrics like spelling rate
        for ERP based experiments.

        Parameters
        ----------
        perf : dict of metrics  
            performance of an evaluation on the dataset.
        op : int
            Fold/subject index to be evaluated.

        Returns
        -------
        dict
            dictionary of metrics keys/values
        """
        pm = {}
        if self.settings.paradigm_metrics:
            for metric in self.settings.paradigm_metrics:
                if metric == "itr":
                    continue # postpone itr calculation after accuracy/spelling rate assignment
                pm[metric] = self.settings.paradigm_metrics[metric](perf["probs"], op, self.dataset) 
        # 
        if "itr" in self.settings.paradigm_metrics:
            if "spelling_rate" in self.settings.paradigm_metrics:
                p = pm["spelling_rate"]
            else:
                p = perf["accuray"]
            pm["itr"] = self.settings.paradigm_metrics["itr"](p, op, self.dataset)        
        return pm     

    def _metrcis_multiple_sequence(self, split_perf, op):
        """Calculate evaluation metrics for ERP paradigm or
        any paradigm that requires mutiple trials

        Parameters
        ----------
        split_perf : dict
            initial performance, contains total probabilities and y_test.
        op : int
            operation index, wether a subject or fold
        Returns
        -------
        split_perf : dict
            a dict of metric values for ERP paradigm following sequence repetition
        """
        perfs = []
        probs = split_perf["probs"].squeeze()
        sequence = self.dataset.paradigm.get_repetitions()
        stimuli  = self.dataset.paradigm.get_stimuli()
        phrase   = phrase_from_dataset(self.dataset, op)
        trials   = len(probs) // len(phrase) # n_char_test 
        for seq in range(1, sequence + 1):
            trials_per_char = stimuli * seq      
            seq_index = [np.arange(i, i+trials_per_char)  for i in range(0, len(probs), trials)]
            seq_index = np.concatenate(seq_index)
            y_test = torch.tensor(split_perf["Y_test"][seq_index].squeeze())
            p_test = torch.tensor(probs[seq_index]) 
            perfs.append(self.learner.model._compute_metrics({}, p_test, y_test))
        
        perfs = aggregate_results(perfs)
        perfs["probs"] = split_perf["probs"]
        perfs["confusion"] = split_perf["confusion"]
        perfs["viz"] = split_perf["viz"]
        return perfs

    @abc.abstractmethod
    def _eval_operation(self, op):
        pass
    
    @abc.abstractmethod
    def _split_set(self):
        pass

    @abc.abstractmethod
    def _phases_partiton(self):
        pass   

    @abc.abstractmethod
    def _total_operations(self):
        pass
    
    @abc.abstractmethod
    def _eval_type(self):
        pass