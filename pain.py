import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

def encode_categorical_columns(input_file, output_file):
    # Read the input file
    df = pd.read_excel(input_file) if input_file.endswith(".xlsx") else pd.read_csv(input_file)

    # Initialize LabelEncoder
    encoder = LabelEncoder()

    # Iterate through columns for encoding
    for column in df.columns:
        if df[column].dtype == "object":  # Check if the column is categorical
            # Drop NaN values before encoding
            non_null_values = df[column].dropna()
            encoded_values = encoder.fit_transform(non_null_values)

            # Assign back to the original dataframe, keeping NaNs as is
            df.loc[non_null_values.index, column] = encoded_values

    # Identify and drop columns with incrementing numbers (like ID or S/N)
    cols_to_drop = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):  # Check if the column is numeric
            diff = df[column].diff().dropna()  # Compute the difference between consecutive rows
            if all(diff == 1):  # Check if all differences are 1
                cols_to_drop.append(column)

    # Drop the identified columns
    df.drop(columns=cols_to_drop, inplace=True)

    # Replace infinite values with NaN and fill missing values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill missing values differently based on column type
    df = df.apply(lambda col: col.fillna(col.mean()) if pd.api.types.is_numeric_dtype(col) else col.fillna(col.mode()[0]), axis=0)

    # Drop constant columns
    df = df.loc[:, (df != df.iloc[0]).any()]

    # Automatically detect the dependent variable ("SalePrice" or last column)
    dependent_variable = "SalePrice" if "SalePrice" in df.columns else df.columns[-1]

    # Extract features (X) and target (y)
    X = df.drop(columns=[dependent_variable]).values
    y = df[dependent_variable].values.reshape(-1, 1)

    # Variance Inflation Factor (VIF) Calculation
    if X.shape[1] > 1:  # Ensure there are enough features for VIF calculation
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.drop(columns=[dependent_variable]).columns
        vif_data["VIF"] = [
            variance_inflation_factor(X, i) for i in range(X.shape[1])
        ]

        # Drop columns with VIF > 10 (you can adjust this threshold as needed)
        high_vif_features = vif_data[vif_data["VIF"] > float("inf")]["feature"].tolist()
        df.drop(columns=high_vif_features, inplace=True)

        print(f"Removed columns with high VIF: {high_vif_features}")
        print(vif_data)
    else:
        print("Not enough features for VIF calculation")

    # Save the modified dataframe to the output file
    if output_file.endswith(".xlsx"):
        df.to_excel(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)

    print(f"File saved successfully to {output_file}")


# Example usage
in_file = r"C:\Users\ANOINTINGTAMUNOWUNAR\Downloads\house-prices-advanced-regression-techniques\train.csv"  # Input file name
out_file = r"C:\Users\ANOINTINGTAMUNOWUNAR\Downloads\adjusted.xlsx"  # Output file name
encode_categorical_columns(in_file, out_file)
