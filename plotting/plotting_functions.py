# plotting_functions.py
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect
import pandas as pd

def ensure_dir(directory):
    """Ensure the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_model_prefix():
    """
    Determines the model prefix dynamically based on the MODEL_NAME environment variable.
    If the variable is not set, defaults to 'general_plots'.
    """
    model_name = os.getenv("MODEL_NAME", "general_plots")
    return os.path.join("plotting", "model_plots", model_name, "training")

def plotting():
    """
    Parses and plots numeric columns in CSV files, saving plots dynamically
    under the appropriate general directory (not model-specific).
    """
    from stationarity.stationarity_tests import parse_time_column

    folders = ['developed', 'developing']
    base_dir = os.path.abspath(os.path.join("plotting", "general_plots"))
    ensure_dir(base_dir)

    for folder in folders:
        freq = 'Q' if folder == 'developed' else 'Y'
        folder_dir = os.path.join(base_dir, folder)
        ensure_dir(folder_dir)
        print(f"\n--- Folder: {folder} (freq={freq}) ---")

        folder_path = os.path.abspath(folder)
        if not os.path.exists(folder_path):
            print(f"Folder '{folder}' does not exist. Skipping.")
            continue

        csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
        if not csv_files:
            print(f"No CSV files found in '{folder}'. Skipping.")
            continue

        for filename in csv_files:
            file_path = os.path.join(folder_path, filename)
            print(f"Plotting each variable from {file_path} ...")

            try:
                df = pd.read_csv(file_path)
                if 'TIME' not in df.columns:
                    print(f"Skipping {filename} - no 'TIME' column found.")
                    continue

                df = parse_time_column(df, 'TIME', freq=freq)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_df = df[numeric_cols]

                for col in numeric_cols:
                    plot_path = os.path.join(folder_dir, f"{os.path.splitext(filename)[0]}_{col}.png")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    numeric_df[col].plot(ax=ax, title=f"{col} - {filename}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Value")
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()

                    print(f"Saved plot: {plot_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def plot_train_val_test_predictions(
    full_data,
    predictions_inverse,
    train_start,
    train_end,
    val_start,
    val_end,
    test_start,
    test_end,
    variable_names
):
    """
    Plots the entire series for each variable, saving plots dynamically
    under the corresponding model folder.
    """
    model_prefix = get_model_prefix()
    ensure_dir(model_prefix)

    for i, var in enumerate(variable_names):
        plot_path = os.path.join(model_prefix, f"{var}_train_val_test.png")
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(full_data[:, i], label=f'Full Data ({var})', color='gray', alpha=0.4)
            plt.plot(range(train_start, train_end), full_data[train_start:train_end, i], label='Training', color='blue')
            plt.plot(range(val_start, val_end), full_data[val_start:val_end, i], label='Validation', color='green')
            plt.plot(range(test_start, test_end), full_data[test_start:test_end, i], label='Test', color='orange')
            plt.plot(range(test_start, test_start + len(predictions_inverse)), predictions_inverse[:, i], label='Predicted (Test)', color='red', linestyle='dashed')
            plt.title(f"{var} - Train/Val/Test & Predictions")
            plt.xlabel("Time Steps")
            plt.ylabel(var)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
        except Exception as e:
            print(f"Error plotting {var}: {e}")

def plot_test_vs_prediction(y_true_inv, y_pred_inv, variable_names):
    """
    Multi-subplot for actual vs predicted in the test set, saving the plot dynamically.
    """
    model_prefix = get_model_prefix()
    ensure_dir(model_prefix)

    plot_path = os.path.join(model_prefix, "test_vs_prediction.png")
    try:
        num_vars = len(variable_names)
        plt.figure(figsize=(10, 4 * num_vars))
        for i, var in enumerate(variable_names):
            plt.subplot(num_vars, 1, i + 1)
            plt.plot(y_true_inv[:, i], label=f"Actual {var}", color='orange')
            plt.plot(y_pred_inv[:, i], label=f"Pred {var}", color='red', linestyle='--')
            plt.title(f"{var} - Test vs. Predicted")
            plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"Error plotting test vs prediction: {e}")
