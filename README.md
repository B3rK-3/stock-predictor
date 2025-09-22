# 📈 Stock Predictor

Stock Predictor is a machine learning pipeline for forecasting stock price movements using historical data. It provides data preprocessing, training, and evaluation workflows for both **individual stock models** and **generalized models** across multiple stocks.

---

## 🚀 Features

* **Flexible Input**: Accepts stock data in `.csv` format with required columns:

  * `timestamp`
  * `open`
  * `close`
  * `high`
  * `low`
* **Data Preprocessing**
  Run preprocessing to transform raw CSVs into optimized `.npz` datasets for model training and evaluation.
* **Model Training**

  * `train_stock_price/` → Train models tailored to individual stock tickers.
  * `train_stock_change/` → Train generalized models across multiple stocks (slower but more comprehensive).
* **Evaluation**
  Prepares separate evaluation datasets to benchmark model accuracy.

---

## 📂 Project Structure

```
├── functions.py            # Helper functions
├── getSTOCKdata.py         # Script for fetching stock data
├── preprocess_data.py      # Preprocess raw CSV into npz datasets
├── train_stock_price/      # Training pipeline for individual stocks
├── train_stock_change/     # Training pipeline for generalized model
├── stock_data.npz          # Training + validation dataset (generated)
├── stock_data_eval.npz     # Evaluation dataset (generated)
└── README.md               # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stock-predictor.git
   cd stock-predictor
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *(You may need to create a `requirements.txt` listing packages like NumPy, pandas, PyTorch, etc.)*

---

## 📊 Usage

### 1. Prepare Your Data

Place your stock CSV file in the project directory. Ensure it has at least:

```
timestamp, open, close, high, low
```

### 2. Preprocess Data

Run:

```bash
python preprocess_data.py
```

This generates:

* `stock_data.npz` → for training/validation
* `stock_data_eval.npz` → for evaluation

### 3. Train Models

* **Individual Stock Model**:

  ```bash
  cd train_stock_price
  python train.py
  ```
* **Generalized Model**:

  ```bash
  cd train_stock_change
  python train.py
  ```

### 4. Evaluate

Use `stock_data_eval.npz` to benchmark model performance.

---

## 📈 Applications

* Predict short-term stock price trends.
* Compare performance between individual vs generalized models.
* Serve as a foundation for financial time-series ML research.

---

## 🛠️ Future Improvements

* Add deep learning architectures (LSTM, CNN-LSTM, Transformers).
* Integrate live stock APIs (e.g., Alpha Vantage, Yahoo Finance).
* Build a dashboard for visualization and interactive predictions.

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
