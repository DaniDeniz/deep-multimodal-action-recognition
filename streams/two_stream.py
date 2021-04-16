from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from custom_layer.weighperclass import WeighPerClass
import tensorflow as tf
import numpy as np


def two_stream(rgb_model, pulses_model, aggregation_layer="weigh_per_class", weights=None, num_classes=10):
    """
    Deep multimodal Two Stream architecture for action recognition using Video and Pulses (PPG) information
    :param rgb_model: tf.keras model for action recognition from video
    :param aggregation_layer: Aggregation layer to fuse information from both models. weigh_per_class or sum
    :param pulses_model: tf.keras model for action recognition from pulses
    :param weights: Numpy saved file of the weights of WeighPerClass layer
    :param num_classes: number of classes
    :return: Deep multimodal Two Stream architecture
    """
    rgb_model = Model(rgb_model.input, rgb_model.get_layer("logits").output, name="rgbi3d_model")
    pulses_model = Model(pulses_model.input, pulses_model.get_layer("logits").output, name="pulses_model")

    input_video = layers.Input(rgb_model.input_shape[1:], name="input_video")
    input_pulses = layers.Input(pulses_model.input_shape[1:], name="input_pulses")

    rgb_model.trainable = False
    pulses_model.trainable = False

    x = rgb_model(input_video, training=False)
    y = pulses_model(input_pulses, training=False)

    y = layers.Concatenate(axis=-1)([y, tf.reshape(x[:, -1], (tf.shape(x)[0], 1))])

    x = layers.Reshape((num_classes, 1), name="reshape_final_1")(x)

    y = layers.Reshape((num_classes, 1), name="reshape_final_2")(y)

    x = layers.Concatenate(axis=-1)([x, y])

    if aggregation_layer == "weigh_per_class":
        x = WeighPerClass(initializer="ones", name="weigh_per_class")(x)
    x = K.sum(x, axis=-1)

    preds = layers.Activation("softmax", name="prediction")(x)

    model = Model([input_video, input_pulses], preds)

    if weights is not None:
        w = np.load(weights)
        model.get_layer("weigh_per_class").set_weights(w)
    return model
