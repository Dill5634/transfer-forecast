import sys
import argparse
from stationarity.stationarity_tests import run_stationarity_tests
from plotting.plotting_functions import plotting
from training.training import train_lstm, train_cnn_lstm, train_gru

def main():
    parser = argparse.ArgumentParser(
        description="Run stationarity tests, generate plots, and train forecasting models."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "cnn_lstm", "gru", "all"],
        default="all",
        help="Specify which model to train: 'lstm', 'cnn_lstm', 'gru', or 'all' (default)."
    )
    args = parser.parse_args()

    # Step 1: Run stationarity tests
    print("Running stationarity tests...")
    run_stationarity_tests()

    # Step 2: Generate general plots
    print("Generating general plots...")
    plotting()

    # Step 3: Train the selected model(s)
    print("Starting training process...")
    if args.model == "all":
        print("\nTraining LSTM model...")
        train_lstm()
        print("\nTraining CNN-LSTM model...")
        train_cnn_lstm()
        print("\nTraining GRU model...")
        train_gru()
    elif args.model == "lstm":
        print("\nTraining LSTM model...")
        train_lstm()
    elif args.model == "cnn_lstm":
        print("\nTraining CNN-LSTM model...")
        train_cnn_lstm()
    elif args.model == "gru":
        print("\nTraining GRU model...")
        train_gru()
    else:
        print("No valid model option selected. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
