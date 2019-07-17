import tensorflow as tf
import numpy as np
def conv(inputs,n_filters,ksize,stride,scope=None, reuse=None, activation_fn=tf.nn.relu,
             initializer=tf.contrib.layers.variance_scaling_initializer(),
             padding='SAME'):

        with tf.variable_scope(scope,reuse=reuse):
            print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
            n_in = inputs.get_shape().as_list()[-1]
            print(n_in)

            weights = tf.get_variable(
              'weights',shape=[ksize,n_in,n_filters],
              initializer=initializer,
              #dtype = inputs.dytpes.base_dtype,
              collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.VARIABLES])

            current_layer = tf.nn.conv1d(inputs, weights, stride, padding=padding)

            biases = tf.get_variable(
              'biases',shape=[n_filters,],
              initializer=tf.zeros_initializer(),
              #dtype = inputs.dytpes.base_dtype,
              collections=[tf.GraphKeys.BIASES, tf.GraphKeys.VARIABLES])

            current_layer = tf.nn.bias_add(current_layer, biases)
            current_layer = activation_fn(current_layer)
            return current_layer
