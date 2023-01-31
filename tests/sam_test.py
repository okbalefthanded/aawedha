from aawedha.models.pytorch.eegnettorch import EEGNetTorchSSVEP
from aawedha.evaluation.single_subject import SingleSubject
from aawedha.io.san_diego import SanDiego
from aawedha.utils.utils import set_seed
from aawedha.io.base import DataSet
import numpy as np
import pytest
import glob
import os


def create_dataset():
    # San Diego SSVEP
    data_folder = 'data/ssvep_san_diego'
    if glob.glob('data/ssvep_san_diego/epoched/*.mat'):
        return None
        
    save_path = f'{data_folder}/epoched'
    if os.path.exists(save_path):
        download = False
    else:
        os.mkdir(data_folder)
        download = True

    t = [0. , 2.]
    
    ds = SanDiego()
    ds.generate_set(load_path=data_folder, 
                epoch=t, 
                save_folder=save_path,
                download=download,
                fname='SanDiego'
                )
    return ds

def load_data():
    fpath   = 'data/ssvep_san_diego/epoched/San_Diego.pkl'
    dataset =  SanDiego().load_set(fpath)   
    return dataset

def make_model(channels, samples, n_classes):
    return EEGNetTorchSSVEP(nb_classes = n_classes, Chans = channels, Samples = samples, 
                   dropoutRate = 0.5, kernLength = samples, F1 = 16, D = 1, F2 = 16, 
                   )


def process_evaluation(evl, model=None):    
    config  = {}
    compile = {'loss': 'sparse_categorical_crossentropy',
               'optimizer': ['Sam', {'base_optimizer':'SGD', 'lr':0.01, 'rho':0.05, 'adaptive': False}],
               'metrics': ['accuracy'],
               'train' : 'sam'
               }
    fit = {'batch': 64,
           'epochs': 5,
           'callbacks': []
           }
    config['compile'] = compile
    config['fit']     = fit
    config['device']  = 'CPU'

    evl.set_model(model=model, model_config=config)
    evl.run_evaluation() 


def test_single_subject():
    # set seeds
    set_seed(31)
    
    # create main data folder
    if not os.path.exists("data"):
        os.mkdir("data")
    
    # load data
    data = create_dataset()
    if not isinstance(data, DataSet):
        data = load_data()
    data.print_shapes()
    subjects, samples, channels, trials = data.epochs.shape
    n_classes = data.get_n_classes()
    # define en evaluation
    evl = SingleSubject(partition=[3,1], dataset=data, verbose=0, engine='pytorch')
    evl.generate_split(nfolds=4, strategy='Stratified')
    
    # set model
    model = make_model(channels, samples, n_classes)
    
    process_evaluation(evl, model=model)
    
    # test value
    performance = evl.score.results
    print(performance)
    np.testing.assert_allclose(performance['accuracy_mean'], 0.09, rtol=0.02)


if __name__ == '__main__':
    pytest.main([__file__])