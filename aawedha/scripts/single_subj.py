'''
 - main script for evaluating a model(s) on a dataset (s)
'''
from aawedha.io.inria_ern import Inria_ERN
from aawedha.evaluation.single_subject import SingleSubject
from aawedha.paradigms.erp import ERP
from aawedha.paradigms.subject import Subject
from aawedha.models.EEGModels import EEGNet
from tensorflow.keras.metrics import AUC

import numpy as np

from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')
# generate set : convert raw EEG to a multisubject dataset:
# RUN ONCE
# aawedha DataSet Obeject:
# X : data tensor : (subjects, samples, channels, trials)
# Y : labels :      (subjects, trials)

'''
data_folder = 'drive/My Drive/data/inria_data'
save_path = 'drive/inria_data/epoched/'
t= [0. , 1.25]
ds = Inria_ERN()
ds.generate_set(load_path=data_folder, epoch=t,
                save_folder=save_path)
'''


# load epoched dataset
f = 'data/comp_IV_2a/epoched/Comp_IV_2a.pkl'

dt = Inria_ERN().load_set(f)
subjects, samples, channels, trials = dt.epochs.shape
n_classes = np.unique(dt.y).size

# subject = 3 # subject number

# Signle Subject evaluation
prt = [2, 1, 1]
evl = SingleSubject(n_subjects=subjects, partition=prt, dataset=dt, verbose=2)
evl.generate_split(nfolds=1)  # nfolds=10

# select a model
evl.model = EEGNet(nb_classes=n_classes, Chans=channels, Samples=samples,
                   dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                   dropoutType='Dropout'
                   )

evl.model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
# train/test model
evl.run_evaluation()

print(f'Results : {evl.results}')

# print(f'Subject N: {subject+1} Acc: {evl.results["acc"]} ')
# print(f' Acc: {evl.results["acc"]} ')

# Visualize learning curves
