# Custom L1 distance layer

import tensorflow as tf
from tensorflow.keras.layers import Layer


class L1Dist(Layer):
    # init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # similarity calc
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)
