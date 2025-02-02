# train_gru.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.gru import gru
from plotting.plotting_functions import plot_train_val_test_predictions, plot_test_vs_prediction
from helpers.helper_functions import calculate_stats


def train_gru():
    """
    Trains a GRU model on the CSV files in the 'developed' directory,
    using GDP, CPI, UNRATE, IR, BOP as features.
    """


    # 1) Configuration
    os.environ["MODEL_NAME"] = "GRU"
    folder_path = "developed"
    variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    seq_length = 1


    gru_units = [112]        
    dropout_rate = 0.0
    epochs = 250
    batch_size = 32
    model_save_name = "GRU_model.h5"


    # 2) Data Loading
    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        print(f"No CSV found in '{folder_path}'.")
        return


    df_list = []
    for f in csv_files:
        path = os.path.join(folder_path, f)
        temp_df = pd.read_csv(path)
        df_list.append(temp_df[variables])
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print("Combined shape:", combined_df.shape)


    # 3) Train/Val/Test Split
    data_arr = combined_df.values
    N = len(data_arr)
    train_end = int(N * 0.7)
    val_end   = int(N * 0.85)
    test_end  = N


    # 4) Scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data_arr[:train_end])
    full_scaled = scaler.transform(data_arr)


    # 5) Build Sequences
    X, y = [], []
    for i in range(len(full_scaled) - seq_length):
        X.append(full_scaled[i : i + seq_length])
        y.append(full_scaled[i + seq_length])

    X = np.array(X)
    y = np.array(y)

    # Indices for train, val, test
    train_size = train_end - seq_length
    val_size   = val_end - seq_length

    X_train, y_train = X[:train_size],       y[:train_size]
    X_val,   y_val   = X[train_size:val_size], y[train_size:val_size]
    X_test,  y_test  = X[val_size:],         y[val_size:]

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape,   "y_val:",   y_val.shape)
    print("X_test: ", X_test.shape,  "y_test:",  y_test.shape)


    # 6) Build & Train GRU Model
    n_features = len(variables)
    model = gru(
        input_size=seq_length,
        n_features=n_features,
        gru_units=gru_units,
        dropout_rate=dropout_rate
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )


    # 7) Evaluate on Test Data
    test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss (MAE) = {test_loss:.6f}")
    print(f"Test MSE        = {test_mse:.6f}")


    # 8) Predictions & Inverse Transform
    y_pred_test = model.predict(X_test)
    y_pred_test_inv = scaler.inverse_transform(y_pred_test)
    y_test_inv      = scaler.inverse_transform(y_test)


    # 9) Calculate Statistics
    stats = calculate_stats(y_test_inv, y_pred_test_inv)
    for i, var in enumerate(variables):
        print(f"\n--- {var} ---")
        print(f"  MSE:  {stats['MSE'][i]:.4f}")
        print(f"  MAE:  {stats['MAE'][i]:.4f}")
        print(f"  RMSE: {stats['RMSE'][i]:.4f}")
        print(f"  MAPE: {stats['MAPE'][i]:.2f}%")
        print(f"  Accuracy: ~ {stats['Accuracy'][i]:.2f}%")


    # 10) Plot Results
    full_data_inv = scaler.inverse_transform(full_scaled)

    # Adjust for offset if seq_length > 1
    train_start = seq_length
    val_start   = train_end
    test_start  = val_end

    plot_train_val_test_predictions(
        full_data=full_data_inv,
        predictions_inverse=y_pred_test_inv,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        variable_names=variables
    )

    plot_test_vs_prediction(y_test_inv, y_pred_test_inv, variables)

    # 11) Save Model
    model.save(model_save_name)
    print(f"\nModel saved as '{model_save_name}'")


if __name__ == "__main__":
    train_gru()
