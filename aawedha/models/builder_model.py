from aawedha.models.pytorch.torchmodel import TorchModel
from aawedha.models.pytorch.torchmodel import SAMTorch
from aawedha.models.pytorch.torch_twa import TwaTrain
from aawedha.models.pytorch.torch_swa import SWA


train_strategy = {
    'regular': TorchModel,
    'swa': SWA,
    'twa': TwaTrain,
    'sam': SAMTorch,
    'asam': None,
    'wasam': None
}

def build_learner(compile_config):
    
    ts = TorchModel() # default
    if 'train' in compile_config:
        strategy = compile_config['train']        
        if isinstance(compile_config['train'], list):
            strategy = compile_config['train'][0]
            params = compile_config['train'][1]
            ts = train_strategy[strategy](**params)
        else:
            ts = train_strategy[strategy]()            
    return ts