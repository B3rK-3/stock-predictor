import requests
import yfinance

df = yfinance.download('A', period='1mo', interval='1h')

df.to_csv("A.csv")

