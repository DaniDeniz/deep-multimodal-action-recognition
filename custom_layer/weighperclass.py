import tensorflow as tf


class WeighPerClass(tf.keras.layers.Layer):
    def __init__(self, initializer="ones", **kwargs):
        super(WeighPerClass, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-2], input_shape[-1]),
            initializer=self.initializer,
            name="kernel",
            trainable=True)

    def get_config(self):
        base_config = super(WeighPerClass, self).get_config()
        base_config["initializer"] = tf.keras.initializers.serialize(self.initializer)
        return base_config

    def call(self, inputs, **kwargs):
        value = tf.keras.layers.Softmax(axis=-1)(self.kernel)
        weighing = tf.math.multiply(inputs, value)
        return weighing
