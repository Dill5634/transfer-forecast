import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape, LSTM, Input, Dropout
import math
import os

def lstm(input_size, n_features, neurons, dropout):
    """
    Creates an LSTM model for multi-output forecasting (all variables at once).
    
    Parameters:
    - input_size (int): The number of timesteps in each sequence.
    - n_features (int): The number of variables (e.g., 5: GDP, CPI, UNRATE, IR, BOP).
    - neurons (list): A list of integers representing the number of LSTM units per layer.
    - dropout (float): Dropout rate for regularization.
    Returns:
    - model (Model): A compiled Keras model that outputs n_features values.
    """
  
    inputs = Input(shape=(input_size, n_features))
    layer = Reshape((input_size, n_features))(inputs)


    for i, n in enumerate(neurons):
        seq = (i < len(neurons) - 1)
        layer = LSTM(n, return_sequences=seq)(layer)
        if dropout > 0:
            layer = Dropout(dropout)(layer)


    outputs = Dense(n_features, activation='linear')(layer)


    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model