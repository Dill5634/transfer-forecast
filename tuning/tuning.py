import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, LSTM, GRU, Dropout, Dense, Conv1D, BatchNormalization, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from models.lstm import lstm
from models.gru import gru
from models.cnn_lstm import cnn_lstm


# LSTM HYPERPARAMETER TUNING
def build_lstm_model(hp):
    """
    Keras Tuner build function for a multi-output LSTM.
    Tunable parameters:
      - units_1: Number of units in the first LSTM layer.
      - units_2: Number of units in the second LSTM layer.
      - dropout_rate: Dropout rate.
    """
    seq_length = 1  
    n_features = 5

    units_1 = hp.Int("units_1", min_value=16, max_value=128, step=16)
    units_2 = hp.Int("units_2", min_value=16, max_value=128, step=16)
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05)

    inputs = Input(shape=(seq_length, n_features))
    x = Reshape((seq_length, n_features))(inputs)
    x = LSTM(units_1, return_sequences=True)(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = LSTM(units_2, return_sequences=False)(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(n_features, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def tune_lstm_hyperparameters():
    """
    Tune hyperparameters for the LSTM model.
    Steps:
      1. Load CSV files from 'developed' folder (using variables GDP, CPI, UNRATE, IR, BOP).
      2. Partition data: 70% train, 15% validation.
      3. Scale data using MinMaxScaler.
      4. Build supervised sequences (seq_length=1).
      5. Run BayesianOptimization tuning.
      6. Retrain best model and save it.
    """
    folder_path = "developed"
    variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    seq_length = 1
    epochs = 250       
    batch_size = 32

    # 1) Gather CSV files and combine data
    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        print(f"No CSV files found in '{folder_path}'.")
        return
    df_list = []
    for f in csv_files:
        path = os.path.join(folder_path, f)
        temp_df = pd.read_csv(path)
        df_list.append(temp_df[variables])
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print("Combined shape:", combined_df.shape)

    # 2) Partition and scale data
    data_arr = combined_df.values
    N = len(data_arr)
    train_end = int(N * 0.7)
    val_end   = int(N * 0.85)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data_arr[:train_end])
    full_scaled = scaler.transform(data_arr)

    # 3) Build sequences
    X, y = [], []
    for i in range(len(full_scaled) - seq_length):
        X.append(full_scaled[i:i+seq_length])
        y.append(full_scaled[i+seq_length])
    X = np.array(X)
    y = np.array(y)
    train_size = train_end - seq_length
    val_size   = val_end - seq_length
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)

    # 4) Define tuner
    directory_name = 'tuner_results'
    project_name = 'lstm_tuning'
    tuner = kt.BayesianOptimization(
        build_lstm_model,
        objective='val_loss',
        max_trials=100,
        executions_per_trial=1,
        directory=directory_name,
        project_name=project_name
    )

    # 5) Run tuner search
    tuner.search(X_train, y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_val, y_val),
                 verbose=1)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters for LSTM:")
    print(best_hp.values)

    # 6) Build, retrain, and save best model
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(X_train, y_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(X_val, y_val),
                   verbose=1)
    model_save_path = os.path.join(directory_name, project_name, "best_lstm_model.h5")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_model.save(model_save_path)
    print(f"Saved best LSTM model as '{model_save_path}'")


# GRU HYPERPARAMETER TUNING
def build_gru_model(hp):
    """
    Build a GRU model for hyperparameter tuning.
    Tunable parameters:
      - num_layers: number of GRU layers.
      - units for each layer.
      - dropout_rate.
    """
    seq_length = 1
    n_features = 5

    num_layers = hp.Int("num_layers", min_value=1, max_value=3, step=1)
    gru_units_list = []
    for i in range(num_layers):
        units = hp.Int(f"units_{i+1}", min_value=16, max_value=256, step=16)
        gru_units_list.append(units)
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05)

    
    model = gru(
        input_size=seq_length,
        n_features=n_features,
        gru_units=gru_units_list,
        dropout=dropout_rate
    )
    return model

def tune_gru_hyperparameters():
    """
    Tune hyperparameters for the GRU model.
    """
    folder_path = "developed"
    variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    seq_length = 1
    epochs = 250
    batch_size = 32

    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        print(f"No CSV files found in '{folder_path}'.")
        return
    df_list = []
    for file in csv_files:
        path = os.path.join(folder_path, file)
        df_list.append(pd.read_csv(path)[variables])
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print("Combined data shape:", combined_df.shape)

    data_arr = combined_df.values
    N = len(data_arr)
    train_end = int(N * 0.7)
    val_end = int(N * 0.85)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data_arr[:train_end])
    full_scaled = scaler.transform(data_arr)

    X, y = [], []
    for i in range(len(full_scaled) - seq_length):
        X.append(full_scaled[i:i+seq_length])
        y.append(full_scaled[i+seq_length])
    X = np.array(X)
    y = np.array(y)
    train_size = train_end - seq_length
    val_size = val_end - seq_length
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)

    directory_name = 'tuner_results'
    project_name = 'gru_tuning'
    tuner = kt.BayesianOptimization(
        build_gru_model,
        objective='val_loss',
        max_trials=200,
        executions_per_trial=1,
        directory=directory_name,
        project_name=project_name
    )

    tuner.search(X_train, y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_val, y_val),
                 verbose=1)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters for GRU:")
    for key, value in best_hp.values.items():
        print(f"  {key}: {value}")

    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(X_train, y_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(X_val, y_val),
                   verbose=1)
    model_save_path = os.path.join(directory_name, project_name, "best_gru_model.h5")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_model.save(model_save_path)
    print(f"Saved best GRU model as '{model_save_path}'")


# CNN-LSTM HYPERPARAMETER TUNING
def build_cnn_lstm_model(hp):
    """
    Build a CNN-LSTM model for hyperparameter tuning.
    Tunable parameters:
      - filters1, filters2 for Conv1D layers.
      - neurons for LSTM layers (fixed at 2 layers here).
      - dropout rate.
      - dense_units in the dense layer.
    """
    seq_length = 1  
    n_features = 5 

    filters1 = hp.Int("filters1", min_value=16, max_value=128, step=16)
    filters2 = hp.Int("filters2", min_value=16, max_value=128, step=16)
    neurons = [hp.Int(f"neurons_{i+1}", min_value=16, max_value=128, step=16) for i in range(2)]
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05)
    dense_units = hp.Int("dense_units", min_value=16, max_value=128, step=16)


    return cnn_lstm(
        seq_length=seq_length,
        n_features=n_features,
        filters1=filters1,
        filters2=filters2,
        kernel_size=1,
        pool_size=1,
        neurons=neurons,
        dropout=dropout,
        dense_units=dense_units
    )

def tune_cnn_lstm_hyperparameters():
    """
    Tune hyperparameters for the CNN-LSTM model.
    Uses StandardScaler for data scaling.
    """
    folder_path = "developed"
    variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    seq_length = 1
    epochs = 250
    batch_size = 32

    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        print(f"No CSV files found in '{folder_path}'.")
        return
    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df_list.append(pd.read_csv(file_path)[variables])
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print("Combined data shape:", combined_df.shape)

    data_arr = combined_df.values
    N = len(data_arr)
    train_end = int(N * 0.7)
    val_end = int(N * 0.85)
    scaler = StandardScaler()
    scaler.fit(data_arr[:train_end])
    full_scaled = scaler.transform(data_arr)

    X, y = [], []
    for i in range(len(full_scaled) - seq_length):
        X.append(full_scaled[i:i+seq_length])
        y.append(full_scaled[i+seq_length])
    X = np.array(X)
    y = np.array(y)
    train_size = train_end - seq_length
    val_size = val_end - seq_length
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)

    directory_name = 'tuner_results'
    project_name = 'cnn_lstm_tuning'
    tuner = kt.BayesianOptimization(
        build_cnn_lstm_model,
        objective='val_loss',
        max_trials=300,
        executions_per_trial=1,
        directory=directory_name,
        project_name=project_name
    )

    tuner.search(X_train, y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_val, y_val),
                 verbose=1)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters for CNN-LSTM:")
    for key, value in best_hps.values.items():
        print(f"  {key}: {value}")

    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(X_train, y_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(X_val, y_val),
                   verbose=1)
    model_save_path = os.path.join(directory_name, project_name, "best_cnn_lstm_model.h5")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_model.save(model_save_path)
    print(f"Saved best CNN-LSTM model as '{model_save_path}'")


# MAIN FUNCTION FOR HYPERPARAMETER TUNING
def main():
    """
    Example usage: python tuning/tuning.py --model_type lstm
    """
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for forecasting models.")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "gru", "cnn_lstm", "all"],
                        help="Specify which model to tune: lstm, gru, cnn_lstm, or all.")
    args = parser.parse_args()

    if args.model_type == "lstm":
        print("Tuning hyperparameters for LSTM model...")
        tune_lstm_hyperparameters()
    elif args.model_type == "gru":
        print("Tuning hyperparameters for GRU model...")
        tune_gru_hyperparameters()
    elif args.model_type == "cnn_lstm":
        print("Tuning hyperparameters for CNN-LSTM model...")
        tune_cnn_lstm_hyperparameters()
    elif args.model_type == "all":
        print("Tuning hyperparameters for all models sequentially...")
        tune_lstm_hyperparameters()
        tune_gru_hyperparameters()
        tune_cnn_lstm_hyperparameters()
    else:
        print("Invalid model type specified.")

if __name__ == "__main__":
    main()
