import pandas as pd 
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('data/data.csv')


encoder = LabelEncoder()

categorical_columns = data.select_dtypes(include=['object']).columns

for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])
        



X = data[['Coffee_Intake','Sleep_Hours','Stress_Level','Physical_Activity_Hours','Age','BMI','Smoking']] 
y= data['Sleep_Quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error:{mean_absolute_error(y_test, y_pred)}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')