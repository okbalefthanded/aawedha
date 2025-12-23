from aawedha.models.pytorch.torch_builders import build_callbacks
from aawedha.paradigms.utils_paradigms import paradigm_metrics


class Settings:

    #pylint: disable=too-many-arguments
    def __init__(self, partition=None, folds=None, 
                 verbose=0, current=0, debug=False):
        self.partition = partition
        self.folds = folds
        self.verbose = verbose
        self.current = current
        self.debug = debug
        self.nfolds = 0
        self.fit_config = {}
        self.paradigm_metrics = {}

    def get_fit_configs(self):
        """Returns fit configurations as tuple

        Returns
        -------
        batch : int
            batch size
        ep : int
            epochs number
        clbks : list
            callbacks added to be watched during training
        """        
        if self.fit_config:
            batch = self.fit_config['batch']
            ep    = self.fit_config['epochs']
            clbks = []
            if 'callbacks' in self.fit_config:
                clbks = self.build_callbacks(self.fit_config['callbacks'])
                self.fit_config['callbacks'] = clbks
            aug = None
            if 'augment' in self.fit_config: 
                aug = self.fit_config['augment']
        else:
            batch = 64
            ep = 300
            clbks = []
            aug = None        
        return batch, ep, clbks, aug 

    def set_paradigm_metrics(self, metrics):
        if not isinstance(metrics, list):
            metrics = [metrics]
        for metric in metrics:
            if isinstance(metric, str):
                self.paradigm_metrics[metric] = paradigm_metrics[metric]
            # TODO
                
    def build_callbacks(self, callbacks_setting):
        """Builds callbacks from settings"""
        callbacks = build_callbacks(callbacks_setting)
        return callbacks
                