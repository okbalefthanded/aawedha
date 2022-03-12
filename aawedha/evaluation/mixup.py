# mixup[1] augmentation technique implementation following Keras.io example
# https://keras.io/examples/vision/mixup/
# [1] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond Empirical Risk Minimization,
#  1–11. Retrieved from http://arxiv.org/abs/1710.09412
from aawedha.utils.evaluation_utils import labels_to_categorical
import tensorflow as tf
import numpy as np

def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    sample_one, labels_one = ds_one
    sample_two, labels_two = ds_two
    batch_size = tf.shape(sample_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both samples and labels by combining a pair of samples/labels
    # (one from each dataset) into one image/label
    samples = sample_one * x_l + sample_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (samples, labels)

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def build_mixup_dataset(X_train, Y_train, X_val, Y_val, aug, batch):
    """Create a TF dataset instance with mixup lambda function. all data labels must be
    converted to categorical before fitting the model.

    Parameters
    ----------
    X_train : ndarray (N_training, channels, samples)
            training data
    X_val : ndarray (N_validation, channels, samples)
            validation data
    X_test : ndarray (N_test, channels, samples)
            test data
    Y_train : 1d array
            training data labels
    Y_val : 1d array
            validation data labels
    Y_test : 1d array 
            test data labels
    aug : str or list
        data augmentation method name is str.
        if list: first elements is the method name and the 2nd holds its parameter.
        eg: for mixup: ["mixup", 0.1], 0.1 is the value of alpha.
    batch : int
        batch size

    Returns
    -------
    X_train
        training data dataset instance with augmented mixup method.
    
    val
        validation data tuple, with X_val dataset instance and Y_val categorical
        labels.
    """
    alpha = 0.2
    if isinstance(aug, list):
        alpha = aug[1]
    X_train = X_train
    Y_train = labels_to_categorical(Y_train)
    val = None
    if isinstance(Y_val, np.ndarray):
        Y_val = labels_to_categorical(Y_val)
        val = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(batch)

    train_ds_one = (tf.data.Dataset.from_tensor_slices((X_train, Y_train))
                            .shuffle(batch * 100)
                            .batch(batch)
                            )
    train_ds_two = (tf.data.Dataset.from_tensor_slices((X_train, Y_train))
                            .shuffle(batch * 100)
                            .batch(batch)
                            )
    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
                
    X_train = train_ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=alpha),
                                       num_parallel_calls=tf.data.AUTOTUNE)
    return X_train, val