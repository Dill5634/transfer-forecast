import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    LSTM, Dropout, Dense, Reshape, MaxPooling1D
)

def cnn_lstm(input_size, n_features, filters1, filters2, kernel_size, pool_size, neurons, dropout, dense_units, use_batchnorm=True, output_size=None):
    """
    Creates a CNN-LSTM model for multi-output time-series forecasting.

    Parameters:
    - input_size (int):    Number of timesteps in each sequence (fixed to 1).
    - n_features (int):    Number of variables at each timestep.
    - filters1 (int):      Number of filters in the first Conv1D layer.
    - filters2 (int):      Number of filters in the second Conv1D layer.
    - kernel_size (int):   Size of the convolution kernel (for both Conv layers).
    - pool_size (int):     Size of the pooling window.
    - neurons (list):      A list of integers for the number of LSTM units per layer.
    - dropout (float):     Dropout rate for regularization in LSTM layers.
    - dense_units (int):   Number of units in the dense layer before output.
    - use_batchnorm (bool):If True, apply batch normalization after each Conv.
    - output_size (int):   Number of outputs (defaults to n_features).

    Returns:
    - model (Model): A compiled Keras model that outputs `output_size` values.
    """
    if output_size is None:
        output_size = n_features

    inputs = Input(shape=(input_size, n_features))

    # Convolutional layers with kernel_size=1 to operate on feature dimension
    x = Conv1D(filters=filters1, kernel_size=1, padding='same', activation='relu')(inputs)
    if use_batchnorm:
        x = BatchNormalization()(x)

    x = Conv1D(filters=filters2, kernel_size=1, padding='same', activation='relu')(x)
    if use_batchnorm:
        x = BatchNormalization()(x)


    #if input_size > pool_size:
        #x = MaxPooling1D(pool_size=pool_size)(x)

    # Reshape for LSTM
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

    # Final output layer
    outputs = Dense(output_size, activation='linear')(x)

    # Model definition
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model
