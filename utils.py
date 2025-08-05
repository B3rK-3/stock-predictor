import pandas as pd
import pandas_ta as pdt
import re
import torch
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from datetime import datetime

l = re.split(r"[\\/]", os.path.abspath(os.getcwd()))
BASE_PATH = "/".join(l[:-1]) + "/"

DATA_PATH = BASE_PATH + "stock-predictor/data"
RESULTS_PATH = BASE_PATH + "stock-predictor/results"
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
        bollinger = pdt.bbands(read['close'], length=20)
        data['bb_width'] = (bollinger['BBU_20_2.0'] - bollinger['BBL_20_2.0']) / bollinger['BBM_20_2.0']

        time_col_name = 'timestamp' if 'timestamp' in read.columns else 'date'
        data['timestamp'] = pd.to_datetime(read[time_col_name])
        
        # 4. Drop all NaNs created by indicators and pct_change
        data.dropna(inplace=True)

        """More Momentum:

    Stochastic Oscillator (%K and %D): Similar to RSI, shows overbought/oversold levels.

    pdt.stoch(data['high'], data['low'], data['close'])

Volatility:

    Bollinger Bands: Specifically, the width of the bands ((Upper - Lower) / Middle) is a great measure of volatility squeeze/expansion.

    bollinger = pdt.bbands(data['close'])

    data['bb_width'] = (bollinger['BBU_20_2.0'] - bollinger['BBL_20_2.0']) / bollinger['BBM_20_2.0']

Volume/Money Flow:

    On-Balance Volume (OBV): A classic indicator that relates price change to volume.

    pdt.obv(data['close'], data['volume'])

Time/Calendar Features:

    data['day_of_week'] = data.index.dayofweek

    data['month'] = data.index.month

    These should be one-hot encoded before being fed to the model."""

        return data
    except FileNotFoundError:
        print(f"Warning: Data file for {stock} not found. Skipping.")
        return None


def prepare_dataset(stock, num_in, num_pred):
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
    feature_columns = [col for col in data.columns if col != 'close_pct' and col != 'timestamp']
    target_column = 'close_pct'
    time = data['timestamp']
    
    X = data[feature_columns]
    y = data[target_column]

    # =================================================================
    # STEP 2: SPLIT INTO FOUR DISTINCT SETS: X_train, X_test, y_train, y_test
    # =================================================================
    train_size = int(len(data) * 0.8)
    
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

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, time_test, len(feature_columns)
