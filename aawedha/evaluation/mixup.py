# mixup[1] augmentation technique implementation following Keras.io example
# https://keras.io/examples/vision/mixup/
# [1] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond Empirical Risk Minimization,
#  1–11. Retrieved from http://arxiv.org/abs/1710.09412
import tensorflow as tf

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