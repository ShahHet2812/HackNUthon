import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import gym
import random

# Load dataset
data = pd.read_csv("inventory_sales_data.csv")
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')  # Ensure chronological order
data.set_index('date', inplace=True)
data = data.asfreq('D')

# Sales Forecasting using ARIMA
def arima_forecast(data, column='sales', order=(2,1,0)):
    model = ARIMA(data[column], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return forecast

# Sales Forecasting using Prophet
def prophet_forecast(data, column='sales'):
    df = data.reset_index()[['date', column]].rename(columns={'date': 'ds', column: 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

# Waste Prediction using Regression
def waste_prediction(data):
    X = data[['sales', 'inventory_level']]
    y = data['waste']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, predictions)}")
    return predictions

# Reinforcement Learning for Inventory Management
class InventoryEnv(gym.Env):
    def __init__(self):
        self.current_stock = 50
        self.max_stock = 100
        self.demand = random.randint(10, 30)
        self.overstock_penalty = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=self.max_stock, shape=(1,), dtype=np.int32)
    
    def step(self, action):
        if action == 0:  # Reduce stock
            self.current_stock -= 10
        elif action == 1:  # Maintain stock
            pass
        elif action == 2:  # Order more stock
            self.current_stock += 20
        
        # Calculate reward (closer to demand is better)
        reward = 1 - (abs(self.current_stock - self.demand) / self.demand)
        
        # Apply penalty if overstocking
        if self.current_stock > 80:
            self.overstock_penalty = (self.current_stock - 80) * 10  # ₹10 per extra unit
            reward -= self.overstock_penalty / 100  # Reduce reward proportionally
        else:
            self.overstock_penalty = 0
        
        done = False
        return np.array([self.current_stock]), reward, done, {}
    
    def reset(self):
        self.current_stock = 50
        return np.array([self.current_stock])

# Running the RL Environment with Human-Readable Output
env = InventoryEnv()
for _ in range(10):
    state = env.reset()
    action = env.action_space.sample()
    new_state, reward, done, _ = env.step(action)
    
    # Human-readable output
    print(f"\n **Current Stock:** {state[0]} units")
    print(f" **Predicted Demand:** {env.demand} units")
    if action == 0:
        print(" **AI Decision:** Reduced stock")
        print(" **Suggestion:** Monitor demand to prevent stockouts")
    elif action == 1:
        print(" **AI Decision:** Maintained stock")
        print(" **Outcome:** Stock is within the optimal range")
    elif action == 2:
        print(" **AI Decision:** Ordered more stock")
        if env.overstock_penalty > 0:
            print(f" **Penalty Applied:** Overstocking detected! Extra cost: ₹{env.overstock_penalty}")
            print(" **Suggestion:** Reduce ordering to avoid wastage")
        else:
            print(" **Outcome:** Stock adjusted optimally")

# Example Usage
arima_forecast(data)
prophet_forecast(data)
waste_prediction(data)