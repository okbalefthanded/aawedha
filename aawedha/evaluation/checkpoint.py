from aawedha.utils.utils import cwd
import pickle
import os


class CheckPoint:
    """Checkpoint class used to save evaluation state for future
    resume after interruption

    this class is a reduced version of Evaluation in terms of attributes

    Attributes
    ----------
    current : int
        current subject (SingleSubject) / fold (CrossSubject) where checkpoint
        is set
        default : 0

    partition : list of 2 or 3 integers
        configuration for data partitioning into train/validation/test subset
        default a 3 integers list of:
            In case of a single dataset without an independent
            Test set
                - (folds/L, folds/M, folds/N) L+M+N = total trials in dataset
                 for SingleSubject evaluation
                - (L_subjects, M_subjects, N_subjects) L+M+N = total subjects
                in dataset for CrossSubject evaluation
            a 2 integers list of:
                In case of a dataset with an independent Test set
                - (folds/L, folds/M) L+M = T total trials in dataset for
                SingleSubject evaluation
                - (L_subjects, M_subjects) L+M = S total subjects in dataset
                 for CrossSubject evaluation

    folds : a list of 3 1d numpy array
        indices of trials(SingleSubject evaluation)/
        subjects(CrossSubjects evaluation) for each fold

    model :  TorchModel instance
        the model to train/test on the dataset

    model_history : list
            history callbacks

    model_config : dict of model configurations, used in compile() and fit().
        compile :
            - loss : str : loss function to optimize during training
                - default  : 'categorical_crossentropy'
            - optimizer : str | optimizer instance : SGD optimizer
                - default : 'adam'
            - metrics : list : str | metrics : training metrics
                - default : multiclass tasks ['accuracy']
                            binary tasks ['accuracy', AUC()]
        fit :
            - batch : int : batch size
                - default : 64
            - epochs : int : training epochs
                - default : 300
            - callbacks : list : model callbacks
                - default : []

    predictions : nd array of predictions
        models output for each example on the dataset :
            - SingleSubject evaluation : subjects x folds x Trials x dim
            - CrossSubject evaluation : folds x Trials x dim

    cm : list
        confusion matrix per fold

    results : dict of evaluation results compiled from models performance
    on the dataset
        - 'acc' : 2d array : Accuracy for each subjects on each folds
        (subjects x folds)
        - 'acc_mean' : double : Accuracy mean over all subjects
        and folds
        - 'acc_mean_per_fold' : 1d array : Accuracy mean per fold over
         all subjects
        - 'acc_mean_per_subj' : 1d array : Accuracy mean per Subject over all
        folds [only for SingleSubject evaluation]
        For binary class tasks :
        - 'auc' : 2d array : AUC for each subjects on each folds
        (subjects x folds)
        - 'auc_mean' : double :  AUC mean over all subjects and folds
        - 'auc_mean_per_fold' :  1d array : AUC mean per fold over
        all subjects
        - 'auc_mean_per_subj' :  AUC mean per Subject over all folds
        [only for SingleSubject evaluation]
        - 'tpr' : 1d array : True positives rate
        - 'fpr' : 1d array : False positives rate

    log : bool
        if True uses logger to log experiment configurations and results,
        default False

    logger : logger


    Methods
    -------
    set_checkpoint

    """

    def __init__(self, evl=None):
        """Constructor
        """
        self.current = None
        self.check_path = None
        self.learner = evl.learner
        self.settings = evl.settings
        self.score = evl.score
        self.log = evl.log
        if evl.logger:
            self.logger = evl.logger.name()
        self.rets = []
        if hasattr(evl, 'mode'):
            self.mode = evl.mode
            self.best_kept = evl.best_kept

    def set_checkpoint(self, current=0, model=None, rets=None):
        """Save evaluation state to resume operations later

        Evaluations instances will be save in a default location inside
        the packages folder as :
        aawedha/checkpoints/current_[Evaluation_Type].pkl
        at resume() the latest saved evaluation will be loaded and resumed

        Parameters
        ----------
        current : int
            current evaluation subject (SingleSubject)/ fold (CrossSubject)
             index

        model: TorchModel instnce
            current trained model to save, using pytorch save() method which
            saves a model in pth format.

        rets: list
            current evaluation results: accuracy, auc, loss,...
            added to keep a single list of all evaluation results for
            further reports.

        Returns
        -------
        no value
        """
        root = cwd()

        if not os.path.isdir(f"{root}/trained"):
            os.mkdir(f"{root}/trained")

        if rets:
            self.rets.append(rets)
        self.current = current
        # save model using built-in save_model, to avoid pickle error
        # HOTFIX FIXME        
        # self.model_name = 'aawedha/trained/current_model.pth'
        self.check_path = f'{root}/trained/current_model.pth'
        # model.save(self.model_name)

        model.save(self.check_path)
        # save evaluation as object?
        save_folder = f'{root}/checkpoints'
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        fname = save_folder + '/' + 'current_' + self.__class__.__name__ + '.pkl'
        print(f'Saving Checkpoint to destination: {fname}')
        with open(fname, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_checkpoint():
        """load saved checkpoint to resume evaluation

        Returns
        -------
        checkpoint :
            a checkpoint of a saved evaluation
        """
        root = cwd()
        # file_name = 'aawedha/checkpoints/current_CheckPoint.pkl'
        file_name = f'{root}/checkpoints/current_CheckPoint.pkl'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                chkpoint = pickle.load(f)
        else:
            raise FileNotFoundError
        return chkpoint

