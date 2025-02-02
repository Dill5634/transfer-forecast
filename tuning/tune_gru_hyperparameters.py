import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import keras_tuner as kt
from models.gru import gru

def build_gru_model(hp):
    """
    Build a GRU model for hyperparameter tuning.

    Tunable parameters:
    - num_layers   : number of GRU layers (1-3)
    - units_i      : number of GRU units in each GRU layer (16-128 in steps of 16)
    - dropout_rate : dropout rate (0-0.5 in steps of 0.05)
    """

    seq_length = 1
    n_features = 5

    # 1) Number of GRU layers
    num_layers = hp.Int("num_layers", min_value=1, max_value=3, step=1)



    # 2) Determine GRU units for each layer
    gru_units_list = []
    for i in range(num_layers):
        units_i = hp.Int(f"units_{i+1}", min_value=16, max_value=256, step=16)
        gru_units_list.append(units_i)

    # 3) Dropout rate
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05)



    # Build the GRU model
    model = gru(
        input_size=seq_length,
        n_features=n_features,
        gru_units=gru_units_list,
        dropout_rate=dropout_rate
    )
    return model

def tune_hyperparameters():
    """
    Hyperparameter tuning for the GRU model using a Bayesian Optimization search.
    1) Load CSV files and combine the variables GDP, CPI, UNRATE, IR, BOP.
    2) Train/validation split (70% / 15%). (The remaining 15% is effectively test.)
    3) Standardize data based on the training split.
    4) Build supervised sequences (seq_length=1).
    5) Use keras_tuner (BayesianOptimization) to find the best hyperparameters.
    6) Retrain the best model.
    7) Save the best model.
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

    # Scale the data
    data_arr = combined_df.values
    N = len(data_arr)
    train_end = int(N * 0.7)
    val_end = int(N * 0.85)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data_arr[:train_end])
    full_scaled = scaler.transform(data_arr)

    # Create sequences for supervised learning (X, y)
    X, y = [], []
    for i in range(len(full_scaled) - seq_length):
        X.append(full_scaled[i:i+seq_length])
        y.append(full_scaled[i+seq_length])
    X = np.array(X)
    y = np.array(y)

    # Split into train and validation
    train_size = train_end - seq_length
    val_size = val_end - seq_length
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)

    # Step 2: Initialize the Keras Tuner
    directory_name = 'tuner_results'
    project_name = 'gru_tuning'

    tuner = kt.BayesianOptimization(
        hypermodel=build_gru_model,
        objective='val_loss',
        max_trials=200,
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
    for hp_key, hp_value in best_hps.values.items():
        print(f"  {hp_key}: {hp_value}")

    # Step 5: Build and train the best model
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Step 6: Save the best model
    model_save_path = os.path.join(directory_name, project_name, "best_gru_model.h5")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_model.save(model_save_path)
    print(f"Best model saved as '{model_save_path}'")

if __name__ == "__main__":
    tune_hyperparameters()
