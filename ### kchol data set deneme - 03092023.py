import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_excel("/Users/turanbuyukkamaci/Downloads/untitled folder/cleaned_data.xlsx", engine="openpyxl")
data['Date'] = pd.to_datetime(data['Date'])

# Convert 'Date' to numeric (days since the start)
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Instantiate and train the Linear Regression model
lri = LinearRegression()
lri.fit(data[['Days']], data['Now'])

# Dates for prediction
future_dates = pd.to_datetime(["01.07.2023", "01.08.2023", "01.09.2023", "01.10.2023",
                              "01.11.2023", "01.12.2023", "01.01.2024", "01.02.2024",
                              "01.03.2024", "01.04.2024", "01.05.2024", "01.06.2024"])

# Convert these dates to the numeric format
future_days = (future_dates - data['Date'].min()).days

# Predict values for these future dates
predictions = lri.predict(future_days.values.reshape(-1, 1))

# Plotting the data
plt.figure(figsize=(15, 7))
sns.lineplot(x='Date', y='Now', data=data, marker='o', label='Actual')
sns.lineplot(x=future_dates, y=predictions, marker='X', color='red', label='Predicted')
plt.title('Monthly Values with Regression Predictions')
plt.ylabel('Value')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
