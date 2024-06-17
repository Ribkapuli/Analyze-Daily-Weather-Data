#Project : Analyze Daily Weather Data :Use a small dataset of daily weather information (temperature, precipitation, etc.)

#Step - 1 : Importing Required Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#Step - 2 : Load the Data

df = pd.read_csv('weather.csv')

#Step - 3 : Data Exploration

print(df.head())
print(df.info())
print(df.describe())

#Step - 4 : Data Visualization

sns.scatterplot(data=df, x='MinTemp', y='MaxTemp', hue='Rainfall')
plt.title('Scatter Plot of MinTemp vs MaxTemp colored by Rainfall')
plt.show()

#Step - 5 : Data Analysis

sns.histplot(df['Humidity9am'], bins=20, kde=True)
plt.title('Distribution of Humidity at 9 am')
plt.xlabel('Humidity at 9 am')
plt.ylabel('Frequency')
plt.show()

# Analyze the relationships between selected features using a scatter plot matrix
selected_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Pressure9am']
sns.pairplot(df[selected_features])
plt.suptitle('Scatter Plot Matrix of Selected Features', y=1.02)
plt.show()

#Step - 6 : Data Visualization(Part 2)

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['MaxTemp'], marker='o')
plt.xlabel('Index')
plt.ylabel('Max Temperature')
plt.title('Max Temperature Over Time')
plt.grid(True)
plt.show()

#Step - 7 : *Advanced Analysis (e.g., predict Rainfall)

# Prepare the data for prediction
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions and calculate the Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')

#Step - 8 : Conclusions and Insights

# Identify the highest and lowest rainfall months
highest_rainfall_index = df['Rainfall'].idxmax()
lowest_rainfall_index = df['Rainfall'].idxmin()
print(f'Highest rainfall index: {highest_rainfall_index}, Lowest rainfall index: {lowest_rainfall_index}')