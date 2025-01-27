# main.py
import sys

from stationarity.stationarity_tests import run_stationarity_tests
from plotting.plotting_functions import plotting
from training.train_lstm import train_lstm

def main():
    # 1) Run stationarity tests
    run_stationarity_tests()

    # 2) Produce some plots (folder-based)
    plotting()

    # 3) Train LSTM
    train_lstm()

if __name__ == "__main__":
    main()
