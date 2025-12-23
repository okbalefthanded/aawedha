from aawedha.models.pytorch.torchmodelext import TorchModelExt
from aawedha.models.pytorch.torchmodel import TorchModel
from aawedha.models.pytorch.torch_sam import SAMTorch
from aawedha.models.pytorch.torch_twa import TwaTrain
from aawedha.models.pytorch.torch_sam import Wasam2
from aawedha.models.pytorch.torch_swa import SWA

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
            params = compile_config['train'][1]
            ts = train_strategy[strategy](**params)
        else:
            ts = train_strategy[strategy]()            
    return ts