from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score , d2_absolute_error_score , mean_squared_error
import pandas as pd 
import numpy as np 


data =pd.read_csv("data/data.csv")

print(data.head())