import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.cnn_lstm import cnn_lstm
from plotting.plotting_functions import plot_train_val_test_predictions, plot_test_vs_prediction
from helpers.helper_functions import calculate_stats

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler


def train_cnn_lstm():
    """
    Trains a CNN-LSTM model for macroeconomic forecasting.
    """
    os.environ["MODEL_NAME"] = "CNN-LSTM"
    folder_path = "developed"
    variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    seq_length = 1

    filters1 = 48
    filters2 = 32
    kernel_size = 7
    pool_size = 3
    neurons = [224, 64]
    dropout = 0.35
    epochs = 300
    batch_size = 32
    model_save_name = "cnn_lstm_model.h5"

    # 1) Collect CSV files
    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        print(f"No CSV found in '{folder_path}'.")
        return

    # 2) Concatenate them
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
    val_end = int(N * 0.85)
    test_end = N 

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_arr[:train_end])
    full_scaled = scaler.transform(data_arr)

    # 4) Build sequences
    X, y = [], []
    for i in range(len(full_scaled) - seq_length):
        X.append(full_scaled[i: i + seq_length])
        y.append(full_scaled[i + seq_length])
    X = np.array(X)
    y = np.array(y)

    # Create time-based indices
    train_size = train_end - seq_length
    val_size = val_end - seq_length
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    X_test, y_test = X[val_size:], y[val_size:]

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:", y_val.shape)
    print("X_test: ", X_test.shape, "y_test:", y_test.shape)

    # 5) Build and train the model
    n_features = len(variables)
    model = cnn_lstm(seq_length, n_features, filters1, filters2, kernel_size, pool_size, neurons, dropout)
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # 6) Evaluate on test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss = {test_loss:.6f}")

    # 7) Predict on test set
    y_pred_test = model.predict(X_test)
    y_pred_test_inv = scaler.inverse_transform(y_pred_test)
    y_test_inv = scaler.inverse_transform(y_test)

    # 8) Calculate stats
    stats = calculate_stats(y_test_inv, y_pred_test_inv)
    for i, var in enumerate(variables):
        print(f"\n--- {var} ---")
        print(f" MSE: {stats['MSE'][i]:.4f}")
        print(f" MAE: {stats['MAE'][i]:.4f}")
        print(f"RMSE: {stats['RMSE'][i]:.4f}")
        print(f"MAPE: {stats['MAPE'][i]:.2f}%")
        print(f"Accuracy ~ {stats['Accuracy'][i]:.2f}%")

    # 9) For full data plotting
    full_data_inv = scaler.inverse_transform(full_scaled)

    # train range = [seq_length..train_end)
    # val   range = [train_end..val_end)
    # test  range = [val_end..test_end)
    train_start = seq_length
    val_start = train_end

    # Plot full data + test predictions
    plot_train_val_test_predictions(
        full_data=full_data_inv,
        predictions_inverse=y_pred_test_inv,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=val_end,
        test_end=test_end,
        variable_names=variables
    )


    # Additionally, plot the test portion in multi-subplot
    plot_test_vs_prediction(y_test_inv, y_pred_test_inv, variables)

    # 10) Save model
    model.save(model_save_name)
    print(f"Model saved as '{model_save_name}'")

if __name__ == "__main__":
    train_cnn_lstm()
