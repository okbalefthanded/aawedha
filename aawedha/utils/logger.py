import logging


levels_names = {
    'debug' : logging.DEBUG,
    'info' : logging.INFO
}

class Logger:

    def __init__(self, fname="logger.log", logger_name="eval_log",
                 level='debug'):
        self.logging_level = levels_names[level]
        self.logger = self._create_logger(fname, logger_name, self.logging_level)
        

    def name(self):
        return self.logger.handlers[0].baseFilename

    def log(self, message):
        if self.logging_level == logging.DEBUG:
            self.logger.debug(message)
        elif self.logging_level == logging.INFO:
            self.logger.info(message)

    def log_results(self, score):
        """Log metrics means after the end of an evaluation to logger"""
        means = [f'{metric}: {v}' for metric,
                 v in score.results.items() if 'mean' in metric]
        # self.logger.debug(' / '.join(means))
        self.log(' / '.join(means))

    def _create_logger(self, fname='logger.log', logger_name='eval_log', 
                       level='debug'):
        """define a logger instance

        Parameters
        ----------
        fname : str
            logger file path
        logger_name : str
            logger name

        Returns
        -------
        logger
            logger instance
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        # Create handlers
        f_handler = logging.FileHandler(fname, mode='a')
        f_handler.setLevel(level)
        # Create formatters and add it to handlers
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        # Add handlers to the logger

        if len(logger.handlers) > 0:
            for hdl in logger.handlers:
                logger.removeHandler(hdl)

        logger.addHandler(f_handler)
        # c_handler = logging.StreamHandler()
        # c_handler.setLevel(logging.WARNING)
        # c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        # c_handler.setFormatter(c_format)
        # logger.addHandler(c_handler)
        return logger
