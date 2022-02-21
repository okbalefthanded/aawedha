# Implements SoftPooling layer
# Reference: 
# Alexandros Stergiou, Ronald Poppe, Grigorios Kalliatakis
# Refining activation downsampling with SoftPool
# https://arxiv.org/abs/2101.00440
# Code taken from:
# https://github.com/qwopqwop200/SoftPool
import tensorflow as tf

#Due to the difference in implementation method, there may be very slight differences in result values.
class SoftPooling1D(tf.keras.layers.Layer):
    def __init__(self,pool_size=2, strides=None, padding='valid',data_format='channels_last'):
        super(SoftPooling1D, self).__init__()
        self.avgpool = tf.keras.layers.AvgPool1D(pool_size,strides,padding,data_format)
    def call(self, x):
        x_exp = tf.math.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool

class SoftPooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = tf.keras.layers.AvgPool2D(pool_size, strides, padding, data_format)
    def call(self, x):
        x_exp = tf.math.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool
    
class SoftPooling3D(tf.keras.layers.Layer):
    def __init__(self,pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None):
        super(SoftPooling3D, self).__init__()
        self.avgpool = tf.keras.layers.AvgPool3D(pool_size,strides,padding,data_format)
    def call(self, x):
        x_exp = tf.math.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool