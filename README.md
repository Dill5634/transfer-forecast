# transfer-forecast

# Transfer Forecast: Forecasting Macroeconomic Indicators with Deep Learning

This project focuses on forecasting macroeconomic indicators using deep learning models (LSTM, GRU, and CNN-LSTM), enhanced with hyperparameter tuning and transfer learning techniques. The models are trained on quarterly UK economic data and then adapted to yearly US data via transfer learning.

---

## 📁 Project Structure

.  
├── developed/                 # UK quarterly macroeconomic datasets (training)  
├── developing/                # US yearly macroeconomic datasets (transfer learning)  
├── helpers/                   # Helper utilities (e.g., metrics, parsing)  
├── models/                    # Model definitions (LSTM, GRU, CNN-LSTM)  
├── plotting/                  # General and model-specific plotting functions  
├── stationarity/              # Stationarity tests (ADF & KPSS)  
├── trained_models/            # Saved models and stats  
├── training/                  # Model training pipeline  
├── transfer_learning/         # Transfer learning logic & saved stats  
├── tuner_results/             # Hyperparameter tuning output  
├── tuning/                    # Keras Tuner tuning scripts  
├── main.py                    # Master script for end-to-end pipeline  
├── requirements.txt           # Project dependencies  
└── README.md                  # You’re here!  

---

## ⚙️ Main Functionalities

1. **Stationarity Testing**  
   Runs ADF and KPSS tests on all datasets.  
2. **Data Plotting**  
   Generates time series plots for all numeric variables.  
3. **Model Training**  
   Trains LSTM, GRU, and CNN-LSTM models using structured datasets.  
4. **Hyperparameter Tuning**  
   Uses Keras Tuner (Bayesian Optimization) to tune each model.  
5. **Transfer Learning**  
   Adapts models trained on UK data to US data using fine-tuning.  
6. **Evaluation & Visualization**  
   Generates plots and metrics (MAE, RMSE, MAPE, accuracy).  

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py --model all
```

Or run a specific model:

```bash
python main.py --model gru
```

### 3. Hyperparameter Tuning

```bash
python tuning/tuning.py --model_type all
```

Or for a specific model:

```bash
python tuning/tuning.py --model_type lstm
```

### 4. Run Transfer Learning

```bash
python transfer_learning/transfer_learning.py --model_type all
```

---

## 📊 Datasets

- Located in `developed/` (quarterly UK data) and `developing/` (yearly US data).  
- Variables used: `GDP`, `CPI`, `UNRATE`, `IR`, `BOP`  
- Files must contain a `TIME` column in `YYYY Qn` or `YYYY` format, depending on frequency.  

---

## 📈 Model Performance

Each trained model evaluates the following metrics:

- **MSE** – Mean Squared Error  
- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Squared Error  
- **MAPE** – Mean Absolute Percentage Error  
- **Accuracy** – Estimated using `100 - MAPE`  

All metrics are saved to `.csv` files and visualized automatically.

---

## 🧠 Models

- **LSTM**: Sequential memory network for time series.  
- **GRU**: Gated Recurrent Unit architecture.  
- **CNN-LSTM**: 1D convolution layers followed by LSTM for mixed feature extraction.  

Each model can be trained, tuned, and evaluated independently.

---

## 📌 Notes

- All output plots are saved to the `plotting/` directory.  
- All saved models and statistics are stored under `trained_models/` and `transfer_learning/`.  

---

## 📬 Contact

For questions or contributions, feel free to open an issue or fork the repo.  
Project maintained by **Dillon Pandya**.
