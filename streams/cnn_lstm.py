from tensorflow.keras import layers
from tensorflow.keras.models import Model


def cnn_lstm(num_classes=9, timesteps=None,
             features=8, pooling=False,
             endpoint_logit=True,
             weights=None):
    """
    1DCNN + LSTM Tensorflow model for action recognition using PPG data (pulse signal)
    :param num_classes: number of actions
    :param timesteps: Number of timesteps to analyze (default None --> variable size)
    :param features: Features size, number of signal points that represents one timestep
    :param pooling: Pooling layers
    :param endpoint_logit: Output raw logits or softmax predictions
    :param weights: pre-trained weights file (hdf5)
    :return: 1DCNN + LSTM Tensorflow model
    """
    input_data = layers.Input((timesteps, features), name="input_data")
    x = layers.Conv1D(16, 7, activation="relu", padding="same", name="conv1d")(input_data)

    x = layers.Conv1D(32, 5, activation="relu", padding="same", name="conv1d_1")(x)
    x = layers.SpatialDropout1D(0.2, name="spatial_dropout1d")(x)
    x = layers.Conv1D(64, 5, activation="relu", padding="same", name="conv1d_2")(x)

    if pooling:
        x = layers.AveragePooling1D(2, 2, padding="same", name="avg_pool1d")(x)

    x = layers.Conv1D(64, 3, activation="relu", padding="same", name="conv1d_3")(x)
    x = layers.SpatialDropout1D(0.2, name="spatial_dropout1d_1")(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same", name="conv1d_4")(x)

    x = layers.Conv1D(128, 3, activation="relu", padding="same", name="conv1d_5")(x)
    x = layers.SpatialDropout1D(0.4, name="spatial_dropout1d_2")(x)
    x = layers.Conv1D(128, 3, activation="relu", padding="same", name="conv1d_6")(x)
    if pooling:
        x = layers.AveragePooling1D(2, 2, padding="same", name="avg_pool1d_1")(x)

    x = layers.Conv1D(128, 3, activation="relu", padding="same", name="conv1d_7")(x)
    x = layers.SpatialDropout1D(0.4, name="spatial_dropout1d_3")(x)
    x = layers.Conv1D(128, 3, activation="relu", padding="same", name="conv1d_8")(x)

    x = layers.LSTM(128, return_sequences=True, recurrent_activation="hard_sigmoid",
                    recurrent_dropout=0.3, dropout=0.3, name="lstm")(x)
    x = layers.LSTM(128, return_sequences=True, recurrent_activation="hard_sigmoid",
                    recurrent_dropout=0.3, dropout=0.3, name="lstm_1")(x)
    x = layers.LSTM(128, return_sequences=False, recurrent_activation="hard_sigmoid",
                    recurrent_dropout=0.3, dropout=0.3, name="lstm_2")(x)

    x = layers.Dense(128, activation="relu", name="dense")(x)
    x = layers.Dropout(0.25, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.25, name="dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.5, name="dropout_2")(x)
    x = layers.Dense(32, activation="relu", name="dense_3")(x)
    x = layers.Dropout(0.5, name="dropout_3")(x)
    x = layers.Dense(num_classes, name="logits")(x)
    if not endpoint_logit:
        x = layers.Activation("softmax", name="prediction")(x)

    model = Model(input_data, x, name="cnn-lstm_pulse")
    if weights is not None:
        model.load_weights(weights, by_name=True)
    return model
