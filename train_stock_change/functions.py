import pandas as pd
import pandas_ta as pdt
import re
import torch
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from datetime import datetime
import random
import torch.nn as nn
from ray import tune

l = re.split(r"[\\/]", os.path.abspath(os.getcwd()))
BASE_PATH = "/".join(l[:-1]) + "/"

DATA_PATH = BASE_PATH + "stock-predictor/data"
RESULTS_PATH = BASE_PATH + "stock-predictor/train_stock_change/results"
os.makedirs(RESULTS_PATH, exist_ok=True)  # Ensure the results directory exists


def load_data(stock):
    """Loads and preprocesses data for a single Active Region (AR)."""
    try:
        read = pd.read_csv(
            f"{DATA_PATH}/{stock}.csv"
        )  # "ticker","name","date","open","high","low","close","adjusted close","volume"
        data = pd.DataFrame()
        # --- Feature Engineering ---
        # 1. Calculate indicators from RAW prices and add as new columns
        data["rsi_14"] = pdt.rsi(read["close"], length=14)
        data["atr"] = pdt.atr(read["high"], read["low"], read["close"], length=14)
        macd_df = pdt.macd(read["close"], fast=12, slow=26, signal=9)
        data["macd"] = macd_df["MACD_12_26_9"]
        data["macd_signal"] = macd_df["MACDs_12_26_9"]
        data["volume_ratio"] = read["volume"] / read["volume"].rolling(30).mean()

        # 2. Calculate pct_change features with NEW names
        data["open_pct"] = read["open"].pct_change()
        data["high_pct"] = read["high"].pct_change()
        data["low_pct"] = read["low"].pct_change()
        data["close_pct"] = read["close"].pct_change()  # This will be our target 'y'
        bollinger = pdt.bbands(read["close"], length=20)
        data["bb_width"] = (
            bollinger["BBU_20_2.0"] - bollinger["BBL_20_2.0"]
        ) / bollinger["BBM_20_2.0"]

        time_col_name = "timestamp" if "timestamp" in read.columns else "date"
        data["timestamp"] = pd.to_datetime(read[time_col_name])

        # 4. Drop all NaNs created by indicators and pct_change
        data.dropna(inplace=True)

        return data
    except FileNotFoundError:
        print(f"Warning: Data file for {stock} not found. Skipping.")
        return None


def prepare_dataset(stock, num_in, num_pred):
    print('hey')
    """
    Loads, processes, splits, and scales data correctly without data leakage,
    then creates sequences for both training and testing.
    """
    # Use the corrected data loading/feature engineering function from our previous discussion
    data = load_data(stock)
    if data is None:
        return None, None, None, None, None, None

    # =================================================================
    # STEP 1: DEFINE FEATURES (X) AND TARGET (y)
    # =================================================================
    feature_columns = [
        col for col in data.columns if col != "close_pct" and col != "timestamp"
    ]
    target_column = "close_pct"
    time = data["timestamp"]

    X = data[feature_columns]
    y = data[target_column]

    # =================================================================
    # STEP 2: SPLIT INTO FOUR DISTINCT SETS: X_train, X_test, y_train, y_test
    # =================================================================
    train_size = int(len(data) * 0.75)

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    time_test = time.iloc[train_size:]

    # =================================================================
    # STEP 3: SCALE FEATURES CORRECTLY (NO DATA LEAKAGE)
    # =================================================================
    # Initialize ONE scaler for all features
    feature_scaler = StandardScaler()

    # Fit the scaler ONLY on the training features (X_train)
    # The scaler learns the mean/std from the training data here
    feature_scaler.fit(X_train)

    # Use the fitted scaler to transform both train and test features
    # This returns NumPy arrays, which is perfect for the next step
    X_train_scaled = feature_scaler.transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Note: y_train and y_test are NOT scaled.

    # =================================================================
    # STEP 4: CREATE SEQUENCES FOR LSTM
    # =================================================================

    # --- Create Training Sequences ---
    x_train_seq, y_train_seq = [], []
    for i in range(len(X_train_scaled) - num_in - num_pred + 1):
        x_train_seq.append(X_train_scaled[i : i + num_in])
        y_train_seq.append(y_train.iloc[i + num_in : i + num_in + num_pred].values)

    # --- Create Testing Sequences ---
    x_test_seq, y_test_seq = [], []
    for i in range(len(X_test_scaled) - num_in - num_pred + 1):
        x_test_seq.append(X_test_scaled[i : i + num_in])
        y_test_seq.append(y_test.iloc[i + num_in : i + num_in + num_pred].values)

    # =================================================================
    # STEP 5: CONVERT TO TENSORS AND RETURN
    # =================================================================
    X_train_tensor = torch.from_numpy(np.array(x_train_seq, dtype=np.float32))
    y_train_tensor = torch.from_numpy(np.array(y_train_seq, dtype=np.float32))
    X_test_tensor = torch.from_numpy(np.array(x_test_seq, dtype=np.float32))
    y_test_tensor = torch.from_numpy(np.array(y_test_seq, dtype=np.float32))

    return (
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        time_test,
        len(feature_columns),
    )


def calculate_metrics(timeline_true, timeline_predicted):
    # Ensure inputs are NumPy arrays for consistency
    timeline_true = np.array(timeline_true)
    timeline_predicted = np.array(timeline_predicted)
    # Calculate Mean Absolute Error (MAE)
    MAE = np.mean(np.abs(timeline_predicted - timeline_true))
    # Calculate Mean Squared Error (MSE)
    MSE = np.mean(np.square(timeline_predicted - timeline_true))
    # Calculate Root Mean Squared Error (RMSE)
    RMSE = np.sqrt(MSE)
    # Calculate Root Mean Squared Logarithmic Error (RMSLE)
    RMSLE = np.sqrt(
        np.mean(np.square(np.log1p(timeline_predicted) - np.log1p(timeline_true)))
    )
    # Calculate R-squared (R²)
    SS_res = np.sum(np.square(timeline_true - timeline_predicted))
    SS_tot = np.sum(np.square(timeline_true - np.mean(timeline_true)))
    R_squared = 1 - (SS_res / SS_tot)
    return MAE, MSE, RMSE, RMSLE, R_squared


class LSTM(nn.Module):
    # __init__ stays the same...
    def __init__(self, input_size, hidden_size, num_layers, output_length, dropout=0.0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.decoder_lstm = nn.LSTM(
            1, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        self.decoder_fc = nn.Linear(hidden_size, 1)

    # The forward pass now accepts the target tensor 'y' for teacher forcing
    def forward(self, x, y=None, teacher_forcing_ratio=0.5):
        # Encoder
        _, (hidden, cell) = self.encoder_lstm(x)

        # Decoder
        decoder_input = torch.zeros(x.size(0), 1, 1).to(x.device)
        outputs = []

        for t in range(self.output_length):
            out_dec, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            out = self.decoder_fc(out_dec)
            outputs.append(out)

            # Decide whether to use teacher forcing for the next step
            use_teacher_forcing = (y is not None) and (
                random.random() < teacher_forcing_ratio
            )

            if use_teacher_forcing:
                # Use the actual ground-truth value as the next input
                decoder_input = y[:, t].unsqueeze(1).unsqueeze(1)
            else:
                # Use the model's own prediction as the next input
                decoder_input = out

        outputs = torch.cat(outputs, dim=1).squeeze(-1)
        return outputs


class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_length, dropout=0.0):
        super().__init__()

        # A single LSTM layer that processes the entire input sequence
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input tensor shape: [batch_size, seq_length, input_size]
            dropout=dropout,
        )

        # A single linear layer to map the final LSTM state to the desired output length
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)   # linear head (no activation)
        )
    def forward(self, x):
        # Pass the input sequence through the LSTM layer
        # We don't need the final hidden and cell states, just the output sequence
        lstm_out, _ = self.lstm(x)

        # We only need the output from the very last time step of the sequence
        # lstm_out has shape [batch_size, seq_length, hidden_size]
        # We take the last time step: lstm_out[:, -1, :]
        last_time_step_out = lstm_out[:, -1, :]

        # Pass the last time step's output to the linear layer to get the final prediction
        prediction = self.fc(last_time_step_out)

        return prediction


class PlateauStopper(tune.stopper.Stopper):
    """Stops trials when the metric has plateaued."""

    def __init__(
        self,
        metric: str,
        min_epochs: int = 20,
        patience: int = 10,
        min_improvement_percent: float = 1e-5,
    ):
        """
        Args:
            metric: The metric to monitor.
            min_epochs: Minimum number of epochs to run before stopping is considered.
            patience: Number of recent epochs to check for improvement.
            min_improvement: The minimum improvement required to not stop the trial.
        """
        self._metrics = metric
        self._min_epochs = min_epochs
        self._patience = patience
        self._min_improvement = min_improvement_percent
        self._trial_history = {}  # To store the history of each trial

    def __call__(self, trial_id: str, result: dict) -> bool:
        """This is called after each tune.report() call."""
        # Initialize history for a new trial
        if trial_id not in self._trial_history:
            self._trial_history[trial_id] = []

        history = self._trial_history[trial_id]
        history.append(result[self._metrics])

        # Don't stop if we haven't reached the minimum number of epochs
        if len(history) <= self._min_epochs:
            return False

        # Check for improvement over the patience window
        # We look at the best value in the last `patience` epochs
        # and compare it to the best value before that window.
        window = history[-self._patience :]
        previous_best = min(history[: -self._patience])
        current_best = min(window)

        # If there's no meaningful improvement, stop the trial
        improvement_needed = previous_best * self._min_improvement / 100
        if previous_best - current_best < improvement_needed:
            print(
                f"Stopping trial {trial_id}: "
                f"No improvement of {improvement_needed} in the last {self._patience} epochs."
            )
            return True

        return False

    def stop_all(self) -> bool:
        """This function is used to stop all trials at once. We don't need it here."""
        return False
