from aawedha.trainers.torchmodelext import TorchModelExt
from aawedha.trainers.torchmodel import TorchModel
from aawedha.trainers.torch_sam import SAMTorch
from aawedha.trainers.torch_twa import TwaTrain
from aawedha.trainers.torch_sam import Wasam2
from aawedha.trainers.torch_swa import SWA

train_strategy = {
    'regular':  TorchModel,
    'extended': TorchModelExt, # mutltple losses
    # TODO: add hybrid generative-discirminative models
    'swa': SWA,
    'twa': TwaTrain,
    'sam': SAMTorch,
    'asam': None,
    'wasam': Wasam2,
   # 'evonorm': ENorm # wrong
}

def build_learner(compile_config):
    
    ts = TorchModel() # default
    if 'train' in compile_config:
        strategy = compile_config['train']        
        # TODO: replace by a dict
        if isinstance(compile_config['train'], list):            
            strategy = compile_config['train'][0]
            params   = compile_config['train'][1]
            ts = train_strategy[strategy](**params)
        else:
            ts = train_strategy[strategy]()            
    return ts