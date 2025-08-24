import requests
import yfinance

stock = 'IWM'
df = yfinance.download(stock, period='60d', interval='15m')

df.to_csv(f"test_data/{stock}.csv")

