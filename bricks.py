import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = yf.download("AAPL", start="2020-01-01", end="2024-01-01", auto_adjust=True)

data['Prediction'] = data['Close'].shift(-1)

data = data.dropna()

X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Prediction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Sample Predictions:", predictions[:10])

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

last_day = X.tail(1)
next_price = model.predict(last_day)
print("Predicted Next Day Price:", next_price)

plt.figure(figsize=(12,6))

plt.plot(data.index, data['Close'], label='Actual Price')

plt.plot(y_test.index, predictions, label='Predicted Price')

plt.title("Stock Price Prediction (AAPL)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()