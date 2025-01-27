# helper_functions.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def calculate_stats(y_true, y_pred):
    """
    Calculate MSE, MAE, RMSE, MAPE, 'Accuracy' for multi-output data.
    Returns a dict of lists, one per variable.
    """
    n_samples, n_vars = y_true.shape
    stats_dict = {
        "MSE": [],
        "MAE": [],
        "RMSE": [],
        "MAPE": [],
        "Accuracy": []
    }
    for v in range(n_vars):
        true_vals = y_true[:, v]
        pred_vals = y_pred[:, v]

        mse_val = mean_squared_error(true_vals, pred_vals)
        mae_val = mean_absolute_error(true_vals, pred_vals)
        rmse_val = np.sqrt(mse_val)
        mape_val = mean_absolute_percentage_error(true_vals, pred_vals) * 100.0
        acc_val  = 100.0 - mape_val

        stats_dict["MSE"].append(mse_val)
        stats_dict["MAE"].append(mae_val)
        stats_dict["RMSE"].append(rmse_val)
        stats_dict["MAPE"].append(mape_val)
        stats_dict["Accuracy"].append(acc_val)
    return stats_dict
