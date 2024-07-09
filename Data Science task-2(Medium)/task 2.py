import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = 'D:\HunarIntern\Data Science task-2(Medium)\house price data.csv'
data = pd.read_csv(file_path)
print(data.head())

# Check for missing values and handle them
data = data.dropna()

# Drop unnecessary columns
data = data.drop(columns=['date', 'street', 'country'])

# Encode categorical variables
categorical_cols = ['city', 'statezip']
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_categorical = encoder.fit_transform(data[categorical_cols])

# Create a DataFrame with the encoded columns
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate the original dataframe with the encoded columns and drop the original categorical columns
data = pd.concat([data.drop(columns=categorical_cols), encoded_df], axis=1)

# Define features and target variable
X = data.drop(columns='price')
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression model with scaled features
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R-squared:", r2)
