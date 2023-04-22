import pickle
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

projectRoot = Path(__file__).parent / "../.."
data = pd.read_csv(projectRoot/"data/processed/customers.csv")


# Normalize Age, Annual Income, and Spending Score using Min-Max Scaling
scaler = MinMaxScaler()
for column in ["Age", "Annual Income ($)", "Spending Score (1-100)", "Work Experience", "Family Size"]:
    data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))

# Feature Selection
selected_columns = ["Gender", "Age", "Annual Income ($)", "Spending Score (1-100)",
                    "Profession", "Work Experience", "Family Size"]
data = data[selected_columns]

# Encoding
data = pd.get_dummies(data, columns=["Gender", "Profession"])

# Outliers Detection and Handling
# Exclude categorical columns from outlier detection
numeric_cols = data.select_dtypes(include=np.number).columns
numeric_data = data[numeric_cols]

# Detect outliers using IQR method
# todo we need to decide what we want to do with the outliers
Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1
outliers = (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))

# b. Winsorize outliers
for column in ["Annual Income ($)"]:
    data[column] = np.where(outliers[column], data[column].clip(lower=Q1[column] - 1.5 *
                            IQR[column], upper=Q3[column] + 1.5 * IQR[column]), data[column])


# Define the target variable and features
X = data.drop(columns=["Spending Score (1-100)"])
y = data["Spending Score (1-100)"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Linear regression model:
# todo (maybe later decision trees, support vector machines or neural networks)

# Train the model
LinReg_model = LinearRegression()
LinReg_model.fit(X_train, y_train)

# Make predictions
y_pred = LinReg_model.predict(X_test)

# Evaluate the LinReg_model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
"""
Mean Squared Error: 0.08
R2 Score: -0.01

Not good."""


# Save the trained model
with open(Path(__file__).parent / "pickled" / "linear_regression_model.pkl", "wb") as file:
    pickle.dump(LinReg_model, file)

# save the scaler
with open(Path(__file__).parent / "pickled" / "linear_regression_scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# save the features
with open(Path(__file__).parent / "pickled" / "linear_regression_features.pkl", "wb") as file:
    pickle.dump(X.columns, file)
