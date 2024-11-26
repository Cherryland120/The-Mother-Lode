import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

def encode_categorical_columns(daf):
    # Read the input file

    # Initialize LabelEncoder
    encoder = LabelEncoder()

    # Iterate through columns for encoding
    for column in daf.columns:
        if daf[column].dtype == "object":  # Check if the column is categorical
            # Drop NaN values before encoding
            non_null_values = daf[column].dropna()
            encoded_values = encoder.fit_transform(non_null_values)

            # Assign back to the original dataframe, keeping NaNs as is
            daf.loc[non_null_values.index, column] = encoded_values

    # Identify and drop columns with incrementing numbers (like ID or S/N)
    cols_to_drop = []
    for column in daf.columns:
        if pd.api.types.is_numeric_dtype(daf[column]):  # Check if the column is numeric
            diff = daf[column].diff().dropna()  # Compute the difference between consecutive rows
            if all(diff == 1):  # Check if all differences are 1
                cols_to_drop.append(column)

    # Drop the identified columns
    daf.drop(columns=cols_to_drop, inplace=True)

    # Replace infinite values with NaN and fill missing values
    daf = daf.replace([np.inf, -np.inf], np.nan)

    # Fill missing values differently based on column type
    daf = daf.apply(lambda col: col.fillna(col.mean()) if pd.api.types.is_numeric_dtype(col) else col.fillna(col.mode()[0]), axis=0)

    # Drop constant columns
    daf = daf.loc[:, (daf != daf.iloc[0]).any()]

    # Automatically detect the dependent variable ("SalePrice" or last column)
    dependent_variable = "SalePrice" if "SalePrice" in daf.columns else daf.columns[-1]

    # Extract features (X) and target (y)
    X = daf.drop(columns=[dependent_variable]).values
    y = daf[dependent_variable].values.reshape(-1, 1)

    # Variance Inflation Factor (VIF) Calculation
    if X.shape[1] > 1:  # Ensure there are enough features for VIF calculation
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.drop(columns=[dependent_variable]).columns
        vif_data["VIF"] = [
            variance_inflation_factor(X, i) for i in range(X.shape[1])
        ]

        # Drop columns with VIF > 10 (you can adjust this threshold as needed)
        high_vif_features = vif_data[vif_data["VIF"] > float("inf")]["feature"].tolist()
        daf.drop(columns=high_vif_features, inplace=True)

        print(f"Removed columns with high VIF: {high_vif_features}")
        print(vif_data)
    else:
        print("Not enough features for VIF calculation")

    # Save the modified dataframe to the output file
        # if output_file.endswith(".xlsx"):
    # df.to_excel(output_file, index=False)
        # else:
    # df.to_csv(output_file, index=False)
    # print(f"File saved successfully to {output_file}")
    return daf

# Example usage
in_file = r"C:\Users\ANOINTINGTAMUNOWUNAR\Downloads\house-prices-advanced-regression-techniques\train.csv"  # Input file name
df = pd.read_excel(in_file) if in_file.endswith(".xlsx") else pd.read_csv(in_file)
df = encode_categorical_columns(df)


# Automatically detect the dependent variable ("SalePrice" or last column)
dependent_variable = "SalePrice" if "SalePrice" in df.columns else df.columns[-1]

# Extract features (X) and target (y)
X = pd.get_dummies(df.drop(columns=[dependent_variable]), drop_first=True).values
Y = df[dependent_variable].values.reshape(-1, 1)

# Hyperparameters
alpha = 0.01  # Learning rate
iterations = 10000  # Number of iterations

# Feature Scaling Function
def feature_scaling(X):
    """
    Normalize the features by standardizing them (zero mean and unit variance).

    Parameters:
    - X: numpy array of features.

    Returns:
    - X_scaled: The scaled (normalized) feature array.
    - X_mean: The mean of each feature.
    - X_std: The standard deviation of each feature.
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8  # Adding a small value to prevent division by zero
    X_scaled = (X - X_mean) / X_std  # Standardize to zero mean and unit variance
    return X_scaled, X_mean, X_std

# Gradient Descent Function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        # Compute prediction
        predictions = X @ theta

        # Compute error
        error = predictions - y

        # Calculate gradient
        gradient = (1 / m) * (X.T @ error)

        # Update parameters
        theta -= alpha * gradient

        # Compute cost (Mean Squared Error)
        cost = (1 / (2 * m)) * np.sum(error**2)
        cost_history.append(cost)

        # Print cost at every 1000 iterations
        if i % 1000 == 0:
            print(f"Iteration {i}, Cost: {cost}")

    return theta, cost_history

# Feature Scaling (Normalization) using the custom function
X, X_mean, X_std = feature_scaling(X)

# Add a bias term (intercept)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add a column of ones for the bias term

# Initialize parameters
theta = np.ones((X.shape[1], 1))  # Initialize theta (weights)

# Run Gradient Descent
theta, cost_history = gradient_descent(X, Y, theta, alpha, iterations)

# Cost Result
print("Cost after optimization:", cost_history[-1])

# Save Parameter Importance
feature_names = ["Bias"] + list(pd.get_dummies(df.drop(columns=[dependent_variable]), drop_first=True).columns)
parameter_importance = pd.DataFrame({
    "Parameter": feature_names,
    "Weight (Theta)": theta.flatten(),
}).sort_values(by="Weight (Theta)", ascending=False)
parameter_importance.to_csv(r"C:\Users\ANOINTINGTAMUNOWUNAR\Downloads\parameter_importance.csv",
                            index=False)
print(f"Parameter importance saved to Downloads")

# =========================
# PREDICTION ON TEST DATASET
# =========================

# Load the test dataset
test_file_path = r"C:\Users\ANOINTINGTAMUNOWUNAR\Downloads\house-prices-advanced-regression-techniques\test.csv"  # Test file path
test_df = pd.read_excel(test_file_path) if test_file_path.endswith(".xlsx") else pd.read_csv(test_file_path)
columns = test_df.columns.tolist()

# Preprocess test data (apply same transformations as training data)
test_X = pd.get_dummies(test_df, drop_first=True).values

# Align test data columns with training data
# Fill missing columns in the test dataset with zeros
train_columns = feature_names[1:]  # Exclude "Bias"
test_columns = pd.get_dummies(test_df, drop_first=True).columns
missing_columns = set(train_columns) - set(test_columns)
for col in missing_columns:
    test_df[col] = 0

# Re-align test data columns to match training order
test_X = test_df[train_columns].values

# Standardize using the same mean and std as training data
test_X = (test_X - X_mean) / X_std

# Add bias term (intercept) to test data
test_X = np.hstack([np.ones((test_X.shape[0], 1)), test_X])  # Add a column of ones

# Make predictions
test_predictions = test_X @ theta
rowstest, coltest = test_X.shape
print(f"For test X {rowstest}, {coltest}")
rowstest, coltest = theta.shape
print(f"For theta {rowstest}, {coltest}")

# Save predictions to the last column of the test dataset
test_df["Predictions"] = test_predictions.flatten()

# Save updated test dataset
output_test_path = r"C:\Users\ANOINTINGTAMUNOWUNAR\Downloads\test_predictions.xlsx"
test_df.to_excel(output_test_path, index=False)
print(f"Predictions saved to {output_test_path}")



# =========================
# Removal of low affecting weights
# =========================

# Convert parameter_importance to DataFrame for filtering
parameter_importance["Abs_Weight"] = parameter_importance["Weight (Theta)"].abs()

# Filter parameters for ±500 range
drop_columns_500 = parameter_importance[
    (parameter_importance["Abs_Weight"] >= -500) & (parameter_importance["Abs_Weight"] <= 500)
]["Parameter"].tolist()
print(f"The dropped columns are {drop_columns_500}")

# Remove identified columns from the dataset
X_df = pd.get_dummies(df.drop(columns=[dependent_variable]), drop_first=True)

# For ±500 range
X_adjusted_500 = X_df.drop(columns=[col for col in drop_columns_500 if col in X_df.columns], errors="ignore")


# Standardize the adjusted data
X_adjusted_500 = (X_adjusted_500 - X_adjusted_500.mean()) / (X_adjusted_500.std() + 1e-8)

# Add bias term to adjusted datasets
X_adjusted_500 = np.hstack([np.ones((X_adjusted_500.shape[0], 1)), X_adjusted_500.values])  # Add bias term

# Initialize theta for adjusted datasets
theta_500 = np.zeros((X_adjusted_500.shape[1], 1))  # Adjust theta size for ±500 range

# Run Gradient Descent on adjusted datasets
theta_500, cost_history_500 = gradient_descent(X_adjusted_500, Y, theta_500, alpha, iterations)

# Output Results
print("Optimized Parameters (Theta):", theta_500.shape)
print("Cost after optimization:", cost_history_500[-1])



# =========================
# Align test data for adjusted models (±500 range)
# =========================
# Ensure the test dataset has the same columns as the training set after dropping the low-importance columns.
test_X_adjusted_500 = test_df.drop(columns=drop_columns_500, errors="ignore")

# Ensure the test dataset columns match the training set
missing_columns_500 = set(train_columns) - set(test_X_adjusted_500.columns)
for col in missing_columns_500:
    test_X_adjusted_500[col] = 0  # Add missing columns with zeros

# Re-order columns to match the training set
test_X_adjusted_500 = test_X_adjusted_500[train_columns].values

# Standardize the adjusted test data using training data's mean and std
test_X_adjusted_500 = (test_X_adjusted_500 - X_mean) / X_std

# Step 1: Get indices of columns to drop
drop_indices = [columns.index(col) for col in drop_columns_500]

# Step 2: Remove those columns from the dataset
test_X_adjusted_500 = np.delete(test_X_adjusted_500, drop_indices, axis=1)

# Add bias term (intercept) to adjusted test data
test_X_adjusted_500 = np.hstack([np.ones((test_X_adjusted_500.shape[0], 1)), test_X_adjusted_500])

# =========================
# Make Predictions with Adjusted Models
# =========================
# For the ±500 range
print(f"Adjusted 500 {test_X_adjusted_500.shape}")
test_predictions2 = test_X_adjusted_500 @ theta_500

# Save predictions to the last column of the test dataset
test_df["Predictions"] = test_predictions2.flatten()

# Save updated test dataset for ±500 predictions
output_test_path_500 = r"C:\Users\ANOINTINGTAMUNOWUNAR\Downloads\test_predictions_500.xlsx"
test_df.to_excel(output_test_path_500, index=False)
print(f"Predictions saved to {output_test_path_500}")

