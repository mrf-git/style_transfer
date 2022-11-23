import tensorflow as tf


class GramLayer(tf.keras.layers.Layer):
    """Layer to compute flattened Gram matrices from input image layers."""

    def call(self, layer_in, **kwargs):
        shape = tf.shape(layer_in)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        num_channels = shape[3]
        num_elems = height * width * num_channels
        gram = tf.reshape(layer_in, (batch_size, -1, num_channels))

        gram = tf.matmul(gram, gram, transpose_a=True)
        gram /= tf.cast(num_elems, gram.dtype)

        return tf.reshape(gram, (batch_size, -1))
