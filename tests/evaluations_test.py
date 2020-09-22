import pytest
import tensorflow as tf
import tensorflow.keras as keras

from aawedha.io.dummy import Dummy
from aawedha.evaluation.cross_subject import CrossSubject
from aawedha.evaluation.single_subject import SingleSubject
import numpy as np
import random


def seed():
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


def make_data():
    data = Dummy(train_shape=(5, 500, 10, 100), test_shape=(5, 500, 10, 50), nb_classes=5)
    subjects, samples, channels, _ = data.epochs.shape
    n_classes = np.unique(data.y[0]).size
    return data, (subjects, samples, channels, n_classes)


def make_model(channels, samples, n_classes):
    return keras.models.Sequential([
        keras.Input(shape=(channels, samples, 1)),
        keras.layers.Conv2D(40, (1, 31)),
        keras.layers.Conv2D(40, (10, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('elu'),
        keras.layers.AveragePooling2D(pool_size=(1, 35), strides=(1, 7)),
        keras.layers.Activation('elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(n_classes, activation='softmax')],
        name="dummy")


def process_evaluation(evl, nfolds=4, strategy='Kfold', model=None):
    if strategy:
        evl.generate_split(nfolds=nfolds, strategy=strategy)
    else:
        evl.generate_split(nfolds=nfolds)
    evl.set_model(model=model)
    evl.run_evaluation()
    return evl.results


def test_single_subject():
    # set seeds
    seed()
    # create random data
    data, shapes = make_data()
    subjects, samples, channels, n_classes = shapes
    # define en evaluation
    evl = SingleSubject(dataset=data, partition=[2, 1], verbose=0)
    # set model
    model = make_model(channels, samples, n_classes)
    results = process_evaluation(evl, nfolds=4, strategy='Stratified', model=model)
    # test value
    assert np.testing.assert_allclose(results['accuracy_mean'], 0.18, rtol=0.2)


def test_cross_subject():
    # set seeds
    seed()
    # create random data
    data, shapes = make_data()
    subjects, samples, channels, n_classes = shapes
    # define en evaluation
    evl = CrossSubject(dataset=data, partition=[4, 1], verbose=0)
    # set model
    model = make_model(channels, samples, n_classes)
    results = process_evaluation(evl, nfolds=1, strategy=None, model=model)
    # test value
    assert np.testing.assert_allclose(results['accuracy_mean'], 0.2, rtol=0.2)


if __name__ == '__main__':
    pytest.main([__file__])