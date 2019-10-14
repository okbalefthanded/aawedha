'''
 - main script for evaluating a model(s) on a dataset (s)
'''


from aawedha.io.physionet_mi import PhysioNet_MI
from aawedha.evaluation.single_subject import SingleSubject
# generate set : convert raw EEG to a multisubject dataset:
# RUN ONCE
# aawedha DataSet Obeject: 
# X : data tensor : (subjects, samples, channels, trials)
# Y : labels :      (subjects, trials) 


data_folder = 'drive/My Drive/data/physiobank/database/eegmmidb/'
save_path = 'drive/My Drive/arl-eegmodels/data/physiobank_mi/epoched/'
t = [[1., 3.], [1.9, 3.9]]

ds = PhysioNet_MI()
ds.generate_set(load_path=data_folder, epoch=t, save_folder=save_path)

# load dataset
file_path = '/content/drive/My Drive/arl-eegmodels/data/physiobank_mi/epoched/physionet_mi.pkl'
ds1 = PhysioNet_MI()
ds1 = ds1.load_set(file_path)

# select an evaluation type: Single-Subject/ Multiple Subjects
subjects = len(ds1.subjects)
prt = [64,15,30]
evl = SingleSubject(n_subjects=subjects, partition=prt, dataset=ds1)
evl.generate_split(nfolds=30)
# select a model

# train model
# evl.run_evaluation()
# evaluate model

# save model