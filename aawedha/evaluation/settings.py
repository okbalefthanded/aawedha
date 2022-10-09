from aawedha.paradigms.utils_paradigms import paradigm_metrics


class Settings:

    #pylint: disable=too-many-arguments
    def __init__(self, partition=None, folds=None, engine="pytorch", 
                 verbose=0, current=0, debug=False):
        self.partition = partition
        self.folds = folds
        self.engine = engine
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
            keras callbacks added to be watched during training
        """
        
        if self.fit_config:
            batch = self.fit_config['batch']
            ep    = self.fit_config['epochs']
            clbks = self.fit_config['callbacks']
            aug = None
            if 'augment' in self.fit_config: 
                aug = self.fit_config['augment']
            # format = self.model_config['fit']['format']
        else:
            batch = 64
            ep = 300
            clbks = []
            aug = None
            # format = 'channels_first'
        
        # K.set_image_data_format(format)
        
        return batch, ep, clbks, aug #, format

    def set_paradigm_metrics(self, metrics):
        if not isinstance(metrics, list):
            metrics = [metrics]
        for metric in metrics:
            if isinstance(metric, str):
                self.paradigm_metrics[metric] = paradigm_metrics[metric]
            # TODO
                