import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    LSTM, Dropout, Dense, Reshape, MaxPooling1D
)

def cnn_lstm(seq_length, n_features, filters1, filters2, kernel_size, pool_size, neurons, dropout, dense_units, use_batchnorm=True, output_size=None):
    """
    Creates a CNN-LSTM model for multi-output time-series forecasting.
    """
    if output_size is None:
        output_size = n_features

    inputs = Input(shape=(seq_length, n_features))

    x = Conv1D(filters=filters1, kernel_size=1, padding='same', activation='relu')(inputs)
    if use_batchnorm:
        x = BatchNormalization()(x)

    x = Conv1D(filters=filters2, kernel_size=1, padding='same', activation='relu')(x)
    if use_batchnorm:
        x = BatchNormalization()(x)

    #if input_size > pool_size:
        #x = MaxPooling1D(pool_size=pool_size)(x)

    x = Reshape((x.shape[1], filters2))(x)

    for i, n in enumerate(neurons):
        return_seq = (i < len(neurons) - 1)
        x = LSTM(n, return_sequences=return_seq)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)


    x = Dense(dense_units, activation='relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    outputs = Dense(output_size, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error',
        metrics=['mae']
    )

    return model
