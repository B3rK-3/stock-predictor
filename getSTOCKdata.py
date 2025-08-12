import requests
import yfinance

df = yfinance.download('AAPL', period='1mo', interval='1h')

df.to_csv("eval_data/AAPL.csv")

