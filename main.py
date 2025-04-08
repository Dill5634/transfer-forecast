import sys
import argparse
from stationarity.stationarity_tests import run_stationarity_tests
from plotting.plotting_functions import plotting
from training.training import train_lstm, train_cnn_lstm, train_gru
from transfer_learning.transfer_learning import transfer_learning  # <-- NEW

def main():
    parser = argparse.ArgumentParser(
        description="Run stationarity tests, generate plots, train models, and perform transfer learning."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "cnn_lstm", "gru", "all"],
        default="all",
        help="Specify which model to use: 'lstm', 'cnn_lstm', 'gru', or 'all' (default)."
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

    # Step 4: Run Transfer Learning
    print("\n=== Starting Transfer Learning ===")
    if args.model == "all":
        for model_type in ["lstm", "cnn_lstm", "gru"]:
            print(f"\n--- Running transfer learning for {model_type.upper()} ---")
            transfer_learning(model_type=model_type)
    else:
        print(f"\n--- Running transfer learning for {args.model.upper()} ---")
        transfer_learning(model_type=args.model)

if __name__ == "__main__":
    main()
