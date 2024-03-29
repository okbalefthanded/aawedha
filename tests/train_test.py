from aawedha.models.pytorch.eegnettorch import EEGNetTorch
from aawedha.io.bci_comp_iv_2a import Comp_IV_2a
from aawedha.evaluation.train import Train
from aawedha.utils.utils import set_seed
import numpy as np
import pytest
import random
import torch
import os

def create_dataset():
    # BCI COMP IV 2a
    data_folder = 'data/Comp_IV_Data_Set_IIa'
    save_path = 'data/Comp_IV_Data_Set_IIa/epoched'
    os.mkdir(data_folder)
    t = [0.5 , 2.5]

    ds = Comp_IV_2a()
    ds.generate_set(load_path=data_folder, 
                epoch=t, 
                save_folder=save_path,
                download=True,
                fname='Comp_IV_2a'
                )

def load_data():
    fpath = 'data/Comp_IV_Data_Set_IIa/epoched/Comp_IV_2a.pkl'
    dataset =  Comp_IV_2a().load_set(fpath)   
    return dataset

def make_model(channels, samples, n_classes):
    return EEGNetTorch(nb_classes = n_classes, Chans = channels, Samples = samples, 
                   dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                   dropoutType = 'Dropout', device='cpu')


def process_evaluation(evl, model=None):    
    config = {}
    compile = {'loss': 'sparse_categorical_crossentropy',
               'optimizer': 'Adam',
               'metrics': ['accuracy']
               }
    fit = {'batch': 32,
           'epochs': 10,
           'callbacks': []
           }
    config['compile'] = compile
    config['fit'] = fit

    config['device'] = 'GPU'

    evl.set_model(model=model, model_config=config)
    evl.run_evaluation(selection=0)    


def test_single_subject():
    # set seeds
    set_seed(42)
    # load data
    data = create_dataset()
    subjects, samples, channels, n_classes = data.epochs.shape
    # define en evaluation
    evl = Train(dataset=data, verbose=0, engine='pytorch')
    # set model
    model = make_model(channels, samples, n_classes)
    process_evaluation(evl, model=model)
    x_test = data.test_epochs[0].squeeze().transpose((2,1,0))
    y_test = data.test_y[0].squeeze() - 1
    performance = evl.model.evaluate(x_test, y_test)
    # test value
    np.testing.assert_allclose(performance['accuracy'], 0.26, rtol=0.2)


if __name__ == '__main__':
    pytest.main([__file__])