import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    LSTM, Dropout, Dense, Reshape, MaxPooling1D
)


def cnn_lstm(input_size, n_features, filters1, filters2, kernel_size, pool_size, neurons, dropout, use_batchnorm=True, dense_units=64):
    """
    Creates a CNN-LSTM model for multi-output time-series forecasting.

    Parameters:
    - input_size (int):    Number of timesteps in each sequence.
    - n_features (int):    Number of variables at each timestep.
    - filters1 (int):      Number of filters in the first Conv1D layer.
    - filters2 (int):      Number of filters in the second Conv1D layer.
    - kernel_size (int):   Size of the convolution kernel (for both Conv layers).
    - pool_size (int):     Size of the pooling window.
    - neurons (list):      A list of integers for the number of LSTM units per layer.
    - dropout (float):     Dropout rate for regularization in LSTM layers.
    - use_batchnorm (bool):If True, apply batch normalization after each Conv.
    - dense_units (int):   Number of units in the dense layer before output.

    Returns:
    - model (Model): A compiled Keras model that outputs n_features values (multi-output).
    """
    inputs = Input(shape=(input_size, n_features))

    # Convolutional layers
    x = Conv1D(filters=filters1, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
    if use_batchnorm:
        x = BatchNormalization()(x)

    x = Conv1D(filters=filters2, kernel_size=kernel_size, padding='same', activation='relu')(x)
    if use_batchnorm:
        x = BatchNormalization()(x)

    if input_size > pool_size:
        x = MaxPooling1D(pool_size=pool_size)(x)

    
    x = Reshape((x.shape[1], filters2))(x)

    # LSTM layers
    for i, n in enumerate(neurons):
        return_seq = (i < len(neurons) - 1) 
        x = LSTM(n, return_sequences=return_seq)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    # Dense layers
    x = Dense(dense_units, activation='relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    outputs = Dense(n_features, activation='linear')(x)

    # Model definition
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error', metrics=['mae'])

    return model