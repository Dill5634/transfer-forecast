# my_project/models/gru.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Reshape, GRU, Dropout, Dense

def gru(input_size, n_features, gru_units, dropout_rate):
    """
    Creates a GRU model for multi-output forecasting using Sequential API.
    
    Parameters
    ----------
    input_size : int.
    n_features : int
    gru_units    : list of int
    dropout_rate    : float
    """
    model = Sequential()
    model.add(Reshape((input_size, n_features), 
                      input_shape=(input_size, n_features)))

    for i, n in enumerate(gru_units):
        return_seq = (i < len(gru_units) - 1)
      
        if i == 0:
            model.add(GRU(n, return_sequences=return_seq))
        else:
            model.add(GRU(n, return_sequences=return_seq))
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(n_features, activation='linear'))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model
