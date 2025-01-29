import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import keras_tuner as kt
from models.cnn_lstm import cnn_lstm

def build_cnn_lstm_model(hp):
    """
    Build a CNN-LSTM model for hyperparameter tuning.

    Tunable parameters:
    - filters1: Number of filters in the first Conv1D layer.
    - filters2: Number of filters in the second Conv1D layer.
    - kernel_size: Kernel size for Conv1D layers.
    - pool_size: Pooling size for MaxPooling1D.
    - neurons: Number of LSTM units in each layer.
    - dropout: Dropout rate for regularization.
    - dense_units: Number of units in the dense layer.
    """
    seq_length = 1  
    n_features = 5 

    # Tuning hyperparameters
    filters1 = hp.Int("filters1", min_value=16, max_value=256, step=16) 
    filters2 = hp.Int("filters2", min_value=16, max_value=256, step=16)
    kernel_size = hp.Choice("kernel_size", [2, 3, 5, 7])
    pool_size = hp.Choice("pool_size", [1, 2, 3])
    neurons = [hp.Int(f"neurons_{i+1}", min_value=16, max_value=256, step=16) for i in range(2)]
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05)
    dense_units = hp.Int("dense_units", min_value=16, max_value=256, step=16)


    return cnn_lstm(
        input_size=seq_length,
        n_features=n_features,
        filters1=filters1,
        filters2=filters2,
        kernel_size=kernel_size,
        pool_size=pool_size,
        neurons=neurons,
        dropout=dropout,
        dense_units=dense_units
    )

def tune_hyperparameters():
    """
    Hyperparameter tuning for the CNN-LSTM model.
    """
    folder_path = "developed"
    variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    seq_length = 1

    epochs = 250
    batch_size = 32

    # Step 1: Load and preprocess data
    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        print(f"No CSV files found in '{folder_path}'.")
        return

    # Combine all CSV data
    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df_list.append(pd.read_csv(file_path)[variables])
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print("Combined data shape:", combined_df.shape)

    # Scale data
    data_arr = combined_df.values
    N = len(data_arr)
    train_end = int(N * 0.7)
    val_end = int(N * 0.85)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_arr[:train_end])
    full_scaled = scaler.transform(data_arr)

    # Create sequences
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

    # Define directory and project name dynamically
    directory_name = 'tuner_results'
    project_name = 'cnn_lstm_tuning'

    # Step 2: Initialize the Keras Tuner
    tuner = kt.BayesianOptimization(
        build_cnn_lstm_model,
        objective=kt.Objective("val_mae", direction="min"),
        max_trials=300,
        executions_per_trial=1,
        directory=directory_name,
        project_name=project_name
    )

    # Step 3: Perform hyperparameter tuning
    tuner.search(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Step 4: Retrieve and print the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters:")
    for hp, value in best_hps.values.items():
        print(f"  {hp}: {value}")

    # Step 5: Train the best model
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Step 6: Save the best model
    model_save_path = os.path.join(directory_name, project_name, "best_cnn_lstm_model.h5")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_model.save(model_save_path)
    print(f"Best model saved as '{model_save_path}'")

if __name__ == "__main__":
    tune_hyperparameters()
