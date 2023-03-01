import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import softmax
class SoftmaxCosineSim(keras.layers.Layer):
    """Custom Keras layer: takes all z-projections as input and calculates
    output matrix which needs to match to [I|O|I|O], where
            I = Unity matrix of size (batch_size x batch_size)
            O = Zero matrix of size (batch_size x batch_size)
    """

    def __init__(self, batch_size, feat_dim, **kwargs):
        super(SoftmaxCosineSim, self).__init__()
        self.batch_size = batch_size
        self.feat_dim = feat_dim
        self.units = (batch_size, 4 * feat_dim)
        self.input_dim = [(None, feat_dim)] * (batch_size * 2)
        self.temperature = 0.2
        self.LARGE_NUM = 1e9

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "batch_size": self.batch_size,
                "feat_dim": self.feat_dim,
                "units": self.units,
                "input_dim": self.input_dim,
                "temperature": self.temperature,
                "LARGE_NUM": self.LARGE_NUM,
            }
        )
        return config

    def call(self, inputs):
        z1 = []
        z2 = []

        for index in range(self.batch_size):
            z1.append(tf.math.l2_normalize(inputs[index][0], -1))
            z2.append(
                tf.math.l2_normalize(inputs[self.batch_size + index][0], -1)
            )
        z1_large = z1
        z2_large = z2

        masks = tf.one_hot(tf.range(self.batch_size), self.batch_size)
        logits_aa = tf.matmul(z1, z1_large, transpose_b=True) / self.temperature
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = tf.matmul(z2, z2_large, transpose_b=True) / self.temperature
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = tf.matmul(z1, z2_large, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(z2, z1_large, transpose_b=True) / self.temperature
        part1 = softmax(tf.concat([logits_ab, logits_aa], 1))
        part2 = softmax(tf.concat([logits_ba, logits_bb], 1))
        output = tf.concat([part1, part2], 1)
        return output
