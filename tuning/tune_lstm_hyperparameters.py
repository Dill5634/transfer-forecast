import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def build_lstm_model(hp):
    """
    Keras Tuner build function for a multi-output LSTM.
    Tunes the number of units in two LSTM layers, the dropout rate,
    and supports multiple LSTM layers.
    
    Parameters Tuned:
    - 'units_1': Number of units in the first LSTM layer.
    - 'units_2': Number of units in the second LSTM layer.
    - 'dropout_rate': Dropout rate for regularization.
    """
    seq_length = 1  
    n_features = 5

    # Tuning hyperparameters
    units_1 = hp.Int("units_1", min_value=32, max_value=128, step=32, default=64)
    units_2 = hp.Int("units_2", min_value=32, max_value=128, step=32, default=64)
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05, default=0.2)

    inputs = Input(shape=(seq_length, n_features))
    layer = Reshape((seq_length, n_features))(inputs)

    # First LSTM layer
    layer = LSTM(units_1, return_sequences=True)(layer)
    if dropout_rate > 0:
        layer = Dropout(dropout_rate)(layer)

    # Second LSTM layer
    layer = LSTM(units_2, return_sequences=False)(layer)
    if dropout_rate > 0:
        layer = Dropout(dropout_rate)(layer)

    # Multi-output final layer
    outputs = Dense(n_features, activation='linear')(layer)

    # Compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    
    return model

def tune_hyperparameters():
    """
    1) Reads CSVs in 'developed' folder, combining the variables GDP,CPI,UNRATE,IR,BOP
    2) Splits dataset: 70% train, 15% val, 15% test (not used in tuner)
    3) Builds sequences (seq_length=1)
    4) Runs Keras Tuner (BayesianOptimization) to find best hyperparams
       (units_1, units_2, dropout_rate)
    5) Retrains best model on train/val again
    6) Saves best model dynamically based on tuner directory and project name
    """
    folder_path = "developed"
    variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    seq_length = 1

    epochs = 200       
    batch_size = 32   

    # Define directory and project name dynamically
    directory_name = 'tuner_results'
    project_name = 'lstm_tuning'

    # 1) Gather CSV files
    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        print(f"No CSV found in '{folder_path}'.")
        return

    # 2) Concatenate
    df_list = []
    for f in csv_files:
        path = os.path.join(folder_path, f)
        temp_df = pd.read_csv(path)
        df_list.append(temp_df[variables])
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print("Combined shape:", combined_df.shape)

    # 3) Partition: train(70%), val(15%), test(15%)
    data_arr = combined_df.values
    N = len(data_arr)
    train_end = int(N * 0.7)
    val_end   = int(N * 0.85)
    
    # Scale on train portion
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data_arr[:train_end])
    full_scaled = scaler.transform(data_arr)

    # 4) Build sequences (X,y)
    X, y = [], []
    for i in range(len(full_scaled) - seq_length):
        X.append(full_scaled[i : i+seq_length])
        y.append(full_scaled[i+seq_length])
    X = np.array(X)
    y = np.array(y)

    train_size = train_end - seq_length
    val_size   = val_end - seq_length
    X_train, y_train = X[:train_size], y[:train_size]
    X_val,   y_val   = X[train_size:val_size], y[train_size:val_size]
    
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape,   "y_val:",   y_val.shape)

    # 5) Define Tuner
    tuner = kt.BayesianOptimization(
        build_lstm_model,  
        objective='val_loss',
        max_trials=100,        
        executions_per_trial=1,  
        directory=directory_name,
        project_name=project_name
    )

    # 6) Perform hyperparameter search
    tuner.search(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # 7) Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters found:")
    print(best_hp.values)

    # 8) Build & retrain best model
    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # 9) Save best model
    model_save_path = os.path.join(directory_name, project_name, "best_lstm_model.h5")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_model.save(model_save_path)
    print(f"Saved best model as '{model_save_path}'")

if __name__ == "__main__":
    tune_hyperparameters()
