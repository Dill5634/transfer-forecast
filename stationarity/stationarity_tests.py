# stationarity_tests.py
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def parse_time_column(df, time_col, freq):
    """
    Converts a 'TIME' column to either quarterly or yearly datetime/period index.
    freq='Q' => parse 'YYYY Qn', freq='Y' => parse 'YYYY'.
    """
    if freq == 'Q':
        def to_quarter_period(val):
            year_str, quarter_str = val.split()
            return pd.Period(year_str + quarter_str, freq='Q')
        df[time_col] = df[time_col].apply(to_quarter_period)
        df.set_index(time_col, inplace=True)
        # Convert to actual Timestamps (end of quarter)
        df.index = df.index.to_timestamp(how='end')
    elif freq == 'Y':
        df[time_col] = df[time_col].astype(str)
        df[time_col] = pd.to_datetime(df[time_col], format='%Y')
        df.set_index(time_col, inplace=True)
    return df

def stationarity_tests(series):
    """
    Returns a dictionary with ADF statistic, p-value, KPSS statistic, and p-value.
    """
    adf_result = adfuller(series.dropna())
    adf_stat = round(adf_result[0], 2)
    adf_pvalue = round(adf_result[1], 2)

    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    kpss_stat = round(kpss_result[0], 2)
    kpss_pvalue = round(kpss_result[1], 2)

    return {
        'ADF Statistic': adf_stat,
        'ADF p-value': adf_pvalue,
        'KPSS Statistic': kpss_stat,
        'KPSS p-value': kpss_pvalue
    }

def run_stationarity_tests():
    """
    Loops over 'developed'(freq=Q) and 'developing'(freq=Y) folders,
    parses CSVs, runs ADF & KPSS stationarity tests, saves results to CSV.
    """
    folders = ['developed','developing']
    all_results = []

    for folder in folders:
        freq = 'Q' if folder.lower()=='developed' else 'Y'
        print(f"\n--- Searching folder: {folder} (freq={freq}) ---")

        csv_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
        if not csv_files:
            print(f"No CSV found in '{folder}'. Skipping.")
            continue

        for csv_file in csv_files:
            file_path = os.path.join(folder, csv_file)
            print(f"Processing {file_path} (freq={freq})...")

            df = pd.read_csv(file_path)
            time_col = 'TIME'
            if time_col not in df.columns:
                print(f"Skipping {csv_file}, no TIME column.")
                continue

            # parse time
            df = parse_time_column(df, time_col, freq=freq)

            # stationarity test for numeric columns
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    res = stationarity_tests(df[col])
                    all_results.append({
                        'Folder': folder,
                        'Filename': csv_file,
                        'Frequency': freq,
                        'Variable': col,
                        **res
                    })

    out_df = pd.DataFrame(all_results)
    out_df.to_csv('stationarity_test_results.csv', index=False)
    print("All results saved to stationarity_test_results.csv")
