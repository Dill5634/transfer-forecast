import sys

from stationarity.stationarity_tests import run_stationarity_tests
from plotting.plotting_functions import plotting
from training.train_lstm import train_lstm
from training.train_cnn_lstm import train_cnn_lstm


def main():
    # 1) Run stationarity tests
    run_stationarity_tests()

    # 2) Produce some plots (folder-based)
    plotting()

    # 3) Train LSTM
    train_lstm()

    # 4) Train CNN-LSTM
    train_cnn_lstm()



if __name__ == "__main__":
    main()
