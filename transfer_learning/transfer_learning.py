import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from plotting.plotting_functions import plot_test_vs_prediction, plot_train_val_test_predictions
from helpers.helper_functions import calculate_stats

def transfer_learning(model_type):
    """
    Transfer Learning Script
    """
    
    os.environ["MODEL_NAME"] = model_type.upper()
    

    model_save_name = model_type.upper() + "_model.h5"
    

    folder_path = "developing"
    variables = ['GDP', 'CPI', 'UNRATE', 'IR', 'BOP']
    seq_length = 1

    # 1) Load data from CSV files in folder_path
    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        print(f"No CSV files found in '{folder_path}'.")
        return
    df_list = []
    for f in csv_files:
        file_path = os.path.join(folder_path, f)
        df_list.append(pd.read_csv(file_path)[variables])
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print("Combined data shape:", combined_df.shape)

    # 2) Scale the data
    data_arr = combined_df.values
    scaler_map = {
        "cnn_lstm": StandardScaler,                
        "lstm": lambda: MinMaxScaler(feature_range=(0, 1)),  
        "gru": lambda: MinMaxScaler(feature_range=(0, 1))      
    }
    scaler_fn = scaler_map.get(model_type.lower(), lambda: MinMaxScaler(feature_range=(0, 1)))
    scaler = scaler_fn()
    scaler.fit(data_arr)
    full_scaled = scaler.transform(data_arr)

    # 3) Build sequences for prediction
    X, y = [], []
    for i in range(len(full_scaled) - seq_length):
        X.append(full_scaled[i: i + seq_length])
        y.append(full_scaled[i + seq_length])
    X = np.array(X)
    y = np.array(y)
    print("Input shape:", X.shape, "Target shape:", y.shape)

    # 4) Load the pre-trained model
    model_base_name = os.path.splitext(model_save_name)[0]
    model_path = os.path.join("trained_models", model_base_name, model_save_name)
    if not os.path.exists(model_path):
        print(f"Trained model file not found at {model_path}")
        return
    model = load_model(model_path)
    print(f"Loaded {model_type.upper()} model from {model_path}")

    # 5) Freeze all layers except the final dense layer
    print("Freezing all layers except the final dense layer...")
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mean_absolute_error',
                  metrics=['mae'])

    # 6) Split data into training, validation, and test sets
    N = len(X)
    train_end = int(N * 0.7)
    val_end = int(N * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # 7) Fine-tune the model on the new data
    fine_tune_epochs = 250
    batch_size = 32
    print(f"\nFine-tuning the {model_type.upper()} model for {fine_tune_epochs} epochs...")
    history = model.fit(X_train, y_train,
                        epochs=fine_tune_epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        verbose=1)

    # 8) Evaluate the fine-tuned model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    if isinstance(test_loss, list):
        test_loss_str = ", ".join([f"{l:.6f}" for l in test_loss])
    else:
        test_loss_str = f"{test_loss:.6f}"
    print(f"\nEvaluation Loss after fine-tuning = {test_loss_str}")

    # 9) Generate predictions and inverse transform the results
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_true_inv = scaler.inverse_transform(y_test)

    # 10) Calculate error statistics
    stats = calculate_stats(y_true_inv, y_pred_inv)
    for i, var in enumerate(variables):
        print(f"\n--- {var} ---")
        print(f" MSE:  {stats['MSE'][i]:.4f}")
        print(f" MAE:  {stats['MAE'][i]:.4f}")
        print(f"RMSE: {stats['RMSE'][i]:.4f}")
        print(f"MAPE: {stats['MAPE'][i]:.2f}%")
        print(f"Accuracy: ~ {stats['Accuracy'][i]:.2f}%")

    # 11) Generate and save the test vs. prediction plot
    transfer_plot_dir = os.path.join("model_plots", model_type.upper(), "transfer")
    os.makedirs(transfer_plot_dir, exist_ok=True)
    plot_test_vs_prediction(y_true_inv, y_pred_inv, variables, subdir="transfer")
    test_plot_path = os.path.join(transfer_plot_dir, "test_vs_prediction.png")
    plt.savefig(test_plot_path)
    plt.clf()
    print(f"Test vs. Prediction plot saved to {test_plot_path}")

    # 12) Generate and save the full dataset plot
    N_full = len(full_scaled)
    train_start_full = seq_length
    train_end_full = int(N_full * 0.7)
    val_start_full = train_end_full
    val_end_full = int(N_full * 0.85)
    test_start_full = val_end_full
    test_end_full = N_full
    full_data_inv = scaler.inverse_transform(full_scaled)
    
    plot_train_val_test_predictions(
        full_data=full_data_inv,
        predictions_inverse=y_pred_inv,
        train_start=train_start_full,
        train_end=train_end_full,
        val_start=val_start_full,
        val_end=val_end_full,
        test_start=test_start_full,
        test_end=test_end_full,
        variable_names=variables,
        subdir="transfer"
    )
    print(f"Full dataset plots saved under {os.path.join('plotting/model_plots', os.getenv('MODEL_NAME'), 'transfer')}")

    # 13) Save statistics to a CSV file in transfer_learning
    transfer_stats_dir = os.path.join("transfer_learning", model_type.upper())
    os.makedirs(transfer_stats_dir, exist_ok=True)
    stats_csv_path = os.path.join(transfer_stats_dir, model_type.upper() +  "_transfer_stats.csv")
    stats_df = pd.DataFrame({
        "Variable": variables,
        "MSE": stats["MSE"],
        "MAE": stats["MAE"],
        "RMSE": stats["RMSE"],
        "MAPE": stats["MAPE"],
        "Accuracy": stats["Accuracy"]
    })
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Transfer learning statistics saved to {stats_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning Script")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["lstm", "gru", "cnn_lstm", "all"],
                        help="Specify which model to run for transfer learning. Use 'all' to run all models sequentially.")
    args = parser.parse_args()

    if args.model_type.lower() == "all":
        for m in ["lstm", "gru", "cnn_lstm"]:
            print(f"\n=== Running transfer learning for {m.upper()} model ===")
            transfer_learning(model_type=m)
    else:
        transfer_learning(model_type=args.model_type)
