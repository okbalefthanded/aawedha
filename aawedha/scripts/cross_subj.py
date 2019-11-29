    '''
    - main script for evaluating a model(s) on a dataset (s)
    '''
    from aawedha.io.physionet_mi import Inria_ERN
    from aawedha.evaluation.single_subject import CrossSubject
    from aawedha.models.EEGModels import EEGNet
    from tf.keras.metrics import AUC
    import numpy as np

    # generate set : convert raw EEG to a multisubject dataset:
    # RUN ONCE
    # aawedha DataSet Obeject:
    # X : data tensor : (subjects, samples, channels, trials)
    # Y : labels :      (subjects, trials)

    data_folder = 'drive/My Drive/data/inria_data'
    save_path = 'drive/inria_data/epoched/'
    t = [0., 1.25]
    ds = Inria_ERN()
    ds.generate_set(load_path=data_folder, epoch=t,
                    save_folder=save_path)

    # load dataset
    file_path = 'data/inria_dataset/epoched/inria_ern.pkl'
    ds1 = Inria_ERN()
    ds1 = ds1.load_set(file_path)

    # select an evaluation type: Multiple Subjects
    subjects = len(ds1.subjects)
    prt = [14, 1, 1]
    evl = CrossSubject(n_subjects=subjects, partition=prt,
                       dataset=ds1)
    evl.generate_split(nfolds=30)
    subjects, samples, channels, trials = evl.dataset.epochs.shape
    n_classes = np.unique(evl.dataset.y).size

    # select a model
    evl.model = EEGNet(nb_classes=n_classes, Chans=channels, Samples=samples,
                       dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                       dropoutType='Dropout')

    evl.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', AUC()]
                      )
    # train model
    evl.run_evaluation()
    # save model
