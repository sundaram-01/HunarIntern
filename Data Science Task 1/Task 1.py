import pandas as pd

# Load dataset
data = pd.read_csv('food_coded.csv')

# Display the first few rows of the dataset
print(data.head())

# Check the dimensions of the dataset
print(f"Dataset shape: {data.shape}")

# Get information about the dataset, including data types and non-null counts
print(data.info())

# Get summary statistics for numerical columns
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)

# Fill missing values for numerical columns with the mean
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[num_cols] = data[num_cols].apply(lambda x: x.fillna(x.mean()), axis=0)

# Fill missing values for categorical columns with the mode
cat_cols = data.select_dtypes(include=['object']).columns
data[cat_cols] = data[cat_cols].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

# Verify that there are no more missing values
print("Missing values after imputation:\n", data.isnull().sum())

# Check for duplicate rows
duplicate_rows = data.duplicated()
print(f"Number of duplicate rows: {duplicate_rows.sum()}")

# Display duplicate rows
print(data[duplicate_rows])

# Remove duplicate rows
data = data.drop_duplicates()

# Verify removal of duplicate rows
print(f"Number of duplicate rows after removal: {data.duplicated().sum()}")

def get_duplicate_columns(df):
    duplicate_column_names = set()
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
        for y in range(x + 1, df.shape[1]):
            other_col = df.iloc[:, y]
            if col.equals(other_col):
                duplicate_column_names.add(df.columns.values[y])
    return list(duplicate_column_names)

# Identify duplicate columns
duplicate_columns = get_duplicate_columns(data)
print(f"Duplicate columns: {duplicate_columns}")

# Remove duplicate columns
data = data.drop(columns=duplicate_columns)

# Verify removal of duplicate columns
print(f"Columns after removing duplicates: {data.columns}")

# Save the cleaned dataset
data.to_csv('cleaned_dataset.csv', index=False)

# Final check
print(data.head())
print(data.info())

