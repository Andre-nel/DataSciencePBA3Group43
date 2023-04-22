
from sklearn.preprocessing import MinMaxScaler


# Normalize Age, Annual Income, and Spending Score using Min-Max Scaling
scaler = MinMaxScaler()
for column in ['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']:
    data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))

# Feature Selection
selected_columns = ['Gender', 'Age',
                    'Annual Income ($)', 'Spending Score (1-100)', 'Profession', 'Work Experience', 'Family Size']
data = data[selected_columns]

# Encoding
data = pd.get_dummies(data, columns=['Gender', 'Profession'])

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
for column in ['Annual Income ($)']:
    data[column] = np.where(outliers[column], data[column].clip(lower=Q1[column] - 1.5 *
                            IQR[column], upper=Q3[column] + 1.5 * IQR[column]), data[column])

# Aggregation

# # a. Aggregate dataset by Profession and Gender
# agg_by_profession_gender = data.groupby(['Profession', 'Gender']).size().reset_index(name='Count')

# # b. Aggregate dataset by Age and Annual Income categories
# agg_by_age_income = data.groupby(['Age', 'Annual Income ($)']).size().reset_index(name='Count')