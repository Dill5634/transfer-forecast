# plotting_functions.py
import matplotlib.pyplot as plt
import numpy as np
import os

def ensure_dir(directory):
    """Ensure the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def plotting():
    """
    Example function that loops over 'developed' and 'developing' to
    parse & plot numeric columns in each CSV, saving the plots.
    """
    import pandas as pd
    from stationarity.stationarity_tests import parse_time_column

    base_dir = os.path.join('plotting', 'model_plots')
    ensure_dir(base_dir)

    folders = ['developed', 'developing']
    for folder in folders:
        freq = 'Q' if folder == 'developed' else 'Y'
        folder_dir = os.path.join(base_dir, folder)
        ensure_dir(folder_dir)
        print(f"\n--- Folder: {folder} (freq={freq}) ---")

        csv_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
        if not csv_files:
            print(f"No CSV in '{folder}'. Skipping.")
            continue

        for filename in csv_files:
            file_path = os.path.join(folder, filename)
            print(f"Processing {file_path} ...")

            df = pd.read_csv(file_path)
            if 'TIME' not in df.columns:
                print(f"Skipping {filename}, no TIME column.")
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
                ax.set_aspect("auto")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

                print(f"Saved plot: {plot_path}")

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
    Plots the entire series for each variable, saving plots for:
      Training, Validation, Test, Predicted
    """
    base_dir = os.path.join('plotting', 'model_plots', 'train_val_test_predictions')
    ensure_dir(base_dir)

    for i, var in enumerate(variable_names):
        plot_path = os.path.join(base_dir, f"{var}_train_val_test.png")
        plt.figure(figsize=(12, 6))
        # Full data
        plt.plot(
            full_data[:, i],
            label=f'Full Data ({var})',
            color='gray',
            alpha=0.4
        )
        # Train portion
        plt.plot(
            range(train_start, train_end),
            full_data[train_start:train_end, i],
            label='Training',
            color='blue'
        )
        # Validation
        plt.plot(
            range(val_start, val_end),
            full_data[val_start:val_end, i],
            label='Validation',
            color='green'
        )
        # Test
        plt.plot(
            range(test_start, test_end),
            full_data[test_start:test_end, i],
            label='Test',
            color='orange'
        )
        # Predicted test
        pred_start = test_start
        pred_end = test_start + len(predictions_inverse)
        plt.plot(
            range(pred_start, pred_end),
            predictions_inverse[:, i],
            label='Predicted (Test)',
            color='red',
            linestyle='dashed'
        )

        plt.title(f"{var} - Train/Val/Test & Predictions")
        plt.xlabel("Time Steps")
        plt.ylabel(var)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved plot: {plot_path}")

def plot_test_vs_prediction(y_true_inv, y_pred_inv, variable_names):
    """
    Multi-subplot for actual vs predicted in test set, saving the plot.
    """
    base_dir = os.path.join('plotting', 'model_plots', 'test_vs_prediction')
    ensure_dir(base_dir)

    plot_path = os.path.join(base_dir, "test_vs_prediction.png")
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