from aawedha.models.pytorch.torchmodel import TorchModel


train_strategy = {
    'regular': TorchModel,
    'swa': None,
    'twa': None,
    'sam': None,
    'asam': None,
    'wasam': None
}

def build_learner(compile_config):
    strategy = 'regular'
    if 'train' in compile_config:
        strategy = compile_config['train']
        if isinstance(compile_config['train'], list):
            strategy =  compile_config['train'][0]
    return train_strategy[strategy]()