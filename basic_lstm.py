from tensorflow.keras.layers import (
    LayerNormalization,
    LSTM,
    Dense,
    Dropout,
    Activation,
    Input,
)
from tensorflow.keras import Model


def basic_lstm():
    inputs = Input(shape=(10, 90))
    x = LayerNormalization(axis=-2)(inputs)
    x = LSTM(units=512, return_sequences=True, activation="relu")(x)
    x = LayerNormalization(axis=-2)(x)
    x = LSTM(units=256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(3)(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="basic_lstm")
    return model
