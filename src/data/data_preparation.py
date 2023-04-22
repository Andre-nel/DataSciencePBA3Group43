import pandas as pd
from pathlib import Path

# Read data from CSV file
projectRoot = Path(__file__).parent / "../.."
data = pd.read_csv(projectRoot/"data/raw/customers.csv")

# Cleaning

# a. Create a new category named 'Unknown Profession' to represent customers without a specified profession
data['Profession'] = data['Profession'].fillna('Unknown Profession')

# b. Replace zeros in Age column with the median value
# data['Age'] = data['Age'].replace(0, data['Age'].median())
# b.
# # Drop rows where work experience is greater than age
data = data.drop(data[data['Work Experience'] > data['Age']].index)

# Drop rows where age is less than 18 and customer has work experience or profession
data = data.drop(data[(data['Age'] < 18) & ((data['Work Experience'] > 0)
                 | (data['Profession'] != 'Unknown Profession'))].index)


# c. Keep zeros in Work Experience column as is, assuming they represent customers with no work experience.
# data['Work Experience'] = data['Work Experience'].replace(0, data['Work Experience'].median())

# Transformation

# a. Bin Age and Annual Income into categories
# age_bins = [0, 17, 29, 44, 59, np.inf]
# age_labels = ['0-17', '18-29', '30-44', '45-59', '60+']
# data['Age'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)

# income_bins = [0, 30000, 60000, np.inf]
# income_labels = ['Low', 'Medium', 'High']
# data['Annual Income ($)'] = pd.cut(data['Annual Income ($)'], bins=income_bins, labels=income_labels)

# Moved to modelling
# Normalize Age, Annual Income, and Spending Score using Min-Max Scaling
# scaler = MinMaxScaler()
# for column in ['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']:
#     data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))

# # Feature Selection
# selected_columns = ['Gender', 'Age',
#                     'Annual Income ($)', 'Spending Score (1-100)', 'Profession', 'Work Experience', 'Family Size']
# data = data[selected_columns]

# # Encoding
# data = pd.get_dummies(data, columns=['Gender', 'Profession'])

# # Outliers Detection and Handling
# # Exclude categorical columns from outlier detection
# numeric_cols = data.select_dtypes(include=np.number).columns
# numeric_data = data[numeric_cols]

# # Detect outliers using IQR method
# # todo we need to decide what we want to do with the outliers
# Q1 = numeric_data.quantile(0.25)
# Q3 = numeric_data.quantile(0.75)
# IQR = Q3 - Q1
# outliers = (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))

# # b. Winsorize outliers
# for column in ['Annual Income ($)']:
#     data[column] = np.where(outliers[column], data[column].clip(lower=Q1[column] - 1.5 *
#                             IQR[column], upper=Q3[column] + 1.5 * IQR[column]), data[column])

# Aggregation

# # a. Aggregate dataset by Profession and Gender
# agg_by_profession_gender = data.groupby(['Profession', 'Gender']).size().reset_index(name='Count')

# # b. Aggregate dataset by Age and Annual Income categories
# agg_by_age_income = data.groupby(['Age', 'Annual Income ($)']).size().reset_index(name='Count')

# Save cleaned and transformed dataset
data.to_csv(projectRoot/"data/processed/customers.csv", index=False)
