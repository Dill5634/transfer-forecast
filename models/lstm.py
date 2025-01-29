# my_project/models/lstm.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, LSTM, Dropout, Dense

def lstm(input_size, n_features, neurons, dropout):
    """
    Creates an LSTM model for multi-output forecasting.
    """
    inputs = Input(shape=(input_size, n_features))
    x = Reshape((input_size, n_features))(inputs)

    for i, n in enumerate(neurons):
        return_seq = (i < len(neurons) - 1)
        x = LSTM(n, return_sequences=return_seq)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    outputs = Dense(n_features, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model
