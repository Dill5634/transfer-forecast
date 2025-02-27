import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from models.lstm import lstm
from models.gru import gru
from models.cnn_lstm import cnn_lstm
from plotting.plotting_functions import plot_train_val_test_predictions, plot_test_vs_prediction
from helpers.helper_functions import calculate_stats, parse_list_of_ints

# 1) LSTM training function
def train_lstm(
    epochs=None,
    batch_size=None,
    neurons=None,
    dropout=None,
    seq_length=None,
    model_save_name=None,
    folder_path="developed",
    variables=None
):
    """
    Trains an LSTM model. Overridden hyperparameters come in as function arguments;
    if they are None, fall back to defaults.
    """
    os.environ["MODEL_NAME"] = "LSTM"

    if variables is None:
        variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    if seq_length is None:
        seq_length = 1
    if neurons is None:
        neurons = [64, 64]
    if dropout is None:
        dropout = 0.0
    if epochs is None:
        epochs = 250
    if batch_size is None:
        batch_size = 32
    if model_save_name is None:
        model_save_name = "LSTM_model.h5"

    training(
        folder_path=folder_path,
        variables=variables,
        seq_length=seq_length,
        model_type="lstm",
        lstm_neurons=neurons,
        lstm_dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        model_save_name=model_save_name
    )


# 2) GRU training function
def train_gru(
    epochs=None,
    batch_size=None,
    gru_units=None,
    gru_dropout_rate=None,
    seq_length=None,
    model_save_name=None,
    folder_path="developed",
    variables=None
):
    """
    Trains a GRU model. Overridden hyperparameters come in as function arguments;
    if they are None, fall back to defaults.
    """
    os.environ["MODEL_NAME"] = "GRU"

    if variables is None:
        variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    if seq_length is None:
        seq_length = 1
    if gru_units is None:
        gru_units = [112]
    if gru_dropout_rate is None:
        gru_dropout_rate = 0.0
    if epochs is None:
        epochs = 250
    if batch_size is None:
        batch_size = 32
    if model_save_name is None:
        model_save_name = "GRU_model.h5"

    training(
        folder_path=folder_path,
        variables=variables,
        seq_length=seq_length,
        model_type="gru",
        gru_units=gru_units,
        gru_dropout_rate=gru_dropout_rate,
        epochs=epochs,
        batch_size=batch_size,
        model_save_name=model_save_name
    )


# 3) CNN-LSTM training function
def train_cnn_lstm(
    epochs=None,
    batch_size=None,
    filters1=None,
    filters2=None,
    kernel_size=None,
    pool_size=None,
    cnn_lstm_neurons=None,
    cnn_lstm_dropout=None,
    dense_units=None,
    seq_length=None,
    model_save_name=None,
    folder_path="developed",
    variables=None
):
    """
    Trains a CNN-LSTM model. Overridden hyperparameters come in as function arguments;
    if they are None, fall back to defaults.
    """
    os.environ["MODEL_NAME"] = "CNN_LSTM"

    if variables is None:
        variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    if seq_length is None:
        seq_length = 1
    if filters1 is None:
        filters1 = 64
    if filters2 is None:
        filters2 = 16
    if kernel_size is None:
        kernel_size = 1
    if pool_size is None:
        pool_size = 1
    if cnn_lstm_neurons is None:
        cnn_lstm_neurons = [16, 48]
    if cnn_lstm_dropout is None:
        cnn_lstm_dropout = 0.4
    if dense_units is None:
        dense_units = 96
    if epochs is None:
        epochs = 300
    if batch_size is None:
        batch_size = 32
    if model_save_name is None:
        model_save_name = "CNN_LSTM_model.h5"

    training(
        folder_path=folder_path,
        variables=variables,
        seq_length=seq_length,
        model_type="cnn_lstm",
        filters1=filters1,
        filters2=filters2,
        kernel_size=kernel_size,
        pool_size=pool_size,
        cnn_lstm_neurons=cnn_lstm_neurons,
        cnn_lstm_dropout=cnn_lstm_dropout,
        dense_units=dense_units,
        epochs=epochs,
        batch_size=batch_size,
        model_save_name=model_save_name
    )


# 4) The training pipeline
def training(
    folder_path,
    variables,
    seq_length,
    model_type,
    epochs,
    batch_size,
    model_save_name,
    # LSTM-specific
    lstm_neurons=None,
    lstm_dropout=None,
    # GRU-specific
    gru_units=None,
    gru_dropout_rate=None,
    # CNN-LSTM-specific
    filters1=None,
    filters2=None,
    kernel_size=None,
    pool_size=None,
    cnn_lstm_neurons=None,
    cnn_lstm_dropout=None,
    dense_units=None
):
    """
    Pipeline for data loading, splitting, scaling, model building,
    training, evaluation, plotting, and saving.
    """
    # 1) Load data from CSV files
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

    # 2) Train/Val/Test Split
    data_arr = combined_df.values
    N = len(data_arr)
    train_end = int(N * 0.7)
    val_end   = int(N * 0.85)
    test_end  = N

    # 3) Scaling
    scaler_map = {
        "cnn_lstm": StandardScaler,
        "lstm": lambda: MinMaxScaler(feature_range=(0, 1)),
        "gru": lambda: MinMaxScaler(feature_range=(0, 1))
    }
    scaler_fn = scaler_map.get(model_type.lower(), lambda: MinMaxScaler(feature_range=(0, 1)))
    scaler = scaler_fn()
    scaler.fit(data_arr)
    full_scaled = scaler.transform(data_arr)

    # 4) Build Sequences
    X, y = [], []
    for i in range(len(full_scaled) - seq_length):
        X.append(full_scaled[i : i + seq_length])
        y.append(full_scaled[i + seq_length])
    X = np.array(X)
    y = np.array(y)

    train_size = train_end - seq_length
    val_size   = val_end - seq_length

    X_train, y_train = X[:train_size], y[:train_size]
    X_val,   y_val   = X[train_size:val_size], y[train_size:val_size]
    X_test,  y_test  = X[val_size:], y[val_size:]
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:",   X_val.shape,   "y_val:",   y_val.shape)
    print("X_test:",  X_test.shape,  "y_test:",  y_test.shape)

    # 5) Build the Model
    n_features = len(variables)

    if model_type.lower() == "lstm":
        # Build LSTM model
        model = lstm(
            seq_length=seq_length,
            n_features=n_features,
            neurons=lstm_neurons,      
            dropout=lstm_dropout       
        )
    elif model_type.lower() == "gru":
        # Build GRU model
        model = gru(
            input_size=seq_length,
            n_features=n_features,
            gru_units=gru_units,       
            dropout=gru_dropout_rate
        )
    elif model_type.lower() == "cnn_lstm":
        model = cnn_lstm(
            seq_length=seq_length,
            n_features=n_features,
            filters1=filters1,        
            filters2=filters2,        
            kernel_size=kernel_size,  
            pool_size=pool_size,      
            neurons=cnn_lstm_neurons,  
            dropout=cnn_lstm_dropout,
            dense_units=dense_units
        )
    else:
        raise ValueError(f"Invalid model_type '{model_type}'. Choose from ['lstm','gru','cnn_lstm'].")

    print(f"\nTraining {model_type.upper()} for {epochs} epochs, batch size {batch_size}...")

    # 6) Train Model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # 7) Evaluate on test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    if isinstance(test_loss, list):
        test_loss_str = ", ".join([f"{loss:.6f}" for loss in test_loss])
    else:
        test_loss_str = f"{test_loss:.6f}"
    print(f"\nTest Loss = {test_loss_str}")

    # 8) Predictions & Inverse Transform
    y_pred_test = model.predict(X_test)
    y_pred_test_inv = scaler.inverse_transform(y_pred_test)
    y_test_inv      = scaler.inverse_transform(y_test)

    # 9) Calculate Statistics
    stats = calculate_stats(y_test_inv, y_pred_test_inv)
    for i, var in enumerate(variables):
        print(f"\n--- {var} ---")
        print(f" MSE:  {stats['MSE'][i]:.4f}")
        print(f" MAE:  {stats['MAE'][i]:.4f}")
        print(f"RMSE: {stats['RMSE'][i]:.4f}")
        print(f"MAPE: {stats['MAPE'][i]:.2f}%")
        print(f"Accuracy: ~ {stats['Accuracy'][i]:.2f}%")

    results_df = pd.DataFrame({
        "Variable": variables,
        "MSE": stats["MSE"],
        "MAE": stats["MAE"],
        "RMSE": stats["RMSE"],
        "MAPE": stats["MAPE"],
        "Accuracy": stats["Accuracy"]
    })

    # 10) Plot Results using subdir="training"
    full_data_inv = scaler.inverse_transform(full_scaled)
    train_start = seq_length
    val_start   = train_end
    test_start  = val_end

    # Use the updated plotting functions with an explicit subdir parameter.
    plot_train_val_test_predictions(
        full_data=full_data_inv,
        predictions_inverse=y_pred_test_inv,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        variable_names=variables,
        subdir="training"
    )
    plot_test_vs_prediction(y_test_inv, y_pred_test_inv, variables, subdir="training")

    # 11) Save Model & Stats
    model_save_folder = "trained_models"
    model_base_name = os.path.splitext(model_save_name)[0]
    model_subfolder = os.path.join(model_save_folder, model_base_name)
    os.makedirs(model_subfolder, exist_ok=True)

    final_model_path = os.path.join(model_subfolder, model_save_name)
    model.save(final_model_path)
    print(f"\nModel saved as '{final_model_path}'")

    stats_csv_path = os.path.join(model_subfolder, f"{model_base_name}_stats.csv")
    results_df.to_csv(stats_csv_path, index=False)
    print(f"Statistics saved as '{stats_csv_path}'")


# 5) Main + Argument Parsing
if __name__ == "__main__":
    """
    Example usage:
      python training/training.py --model_type lstm --epochs 300 --lstm_neurons 128,64 --lstm_dropout 0.2
      python training/training.py --model_type gru --gru_units 128,64 --gru_dropout_rate 0.1
      python training/training.py --model_type cnn_lstm --filters1 32 --filters2 8 --cnn_lstm_neurons 64,32
    If an argument is not provided, it uses the function's hard-coded default.
    """
    parser = argparse.ArgumentParser()
    
    # Model choice
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["lstm", "gru", "cnn_lstm"],
                        help="Choose which model to train. Default = lstm")
    
    # Common overrides
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (override). If None, use defaults.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (override). If None, use defaults.")
    
    # LSTM-specific CLI arguments
    parser.add_argument("--lstm_neurons", type=str, default=None,
                        help="Comma-separated LSTM neurons, e.g. '64,32'.")
    parser.add_argument("--lstm_dropout", type=float, default=None,
                        help="Dropout rate for LSTM, e.g. 0.2.")
    
    # GRU-specific CLI arguments
    parser.add_argument("--gru_units", type=str, default=None,
                        help="Comma-separated GRU units, e.g. '112,64'.")
    parser.add_argument("--gru_dropout_rate", type=float, default=None,
                        help="Dropout rate for GRU, e.g. 0.1.")
    
    # CNN-LSTM-specific CLI arguments
    parser.add_argument("--filters1", type=int, default=None,
                        help="Number of filters in first CNN layer.")
    parser.add_argument("--filters2", type=int, default=None,
                        help="Number of filters in second CNN layer.")
    parser.add_argument("--kernel_size", type=int, default=None,
                        help="Kernel size for CNN layers.")
    parser.add_argument("--pool_size", type=int, default=None,
                        help="Pool size for CNN.")
    parser.add_argument("--cnn_lstm_neurons", type=str, default=None,
                        help="Comma-separated LSTM layer sizes, e.g. '16,48'.")
    parser.add_argument("--cnn_lstm_dropout", type=float, default=None,
                        help="Dropout rate for CNN-LSTM part, e.g. 0.3.")
    parser.add_argument("--dense_units", type=int, default=None,
                        help="Units in final dense layer for CNN-LSTM.")
    
    args = parser.parse_args()
    
    if args.model_type == "lstm":
        train_lstm(
            epochs=args.epochs,
            batch_size=args.batch_size,
            neurons=parse_list_of_ints(args.lstm_neurons),
            dropout=args.lstm_dropout
        )
    elif args.model_type == "gru":
        train_gru(
            epochs=args.epochs,
            batch_size=args.batch_size,
            gru_units=parse_list_of_ints(args.gru_units),
            gru_dropout_rate=args.gru_dropout_rate
        )
    elif args.model_type == "cnn_lstm":
        train_cnn_lstm(
            epochs=args.epochs,
            batch_size=args.batch_size,
            filters1=args.filters1,
            filters2=args.filters2,
            kernel_size=args.kernel_size,
            pool_size=args.pool_size,
            cnn_lstm_neurons=parse_list_of_ints(args.cnn_lstm_neurons),
            cnn_lstm_dropout=args.cnn_lstm_dropout,
            dense_units=args.dense_units
        )
