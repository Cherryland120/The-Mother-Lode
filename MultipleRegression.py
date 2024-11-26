import pandas as pd

# Get the dependent variable name
dvar_name = input("What is the variable you are trying to predict: ")


# Initialize a list to store rows of data
data = []

# Get the number of independent variables and rows
ivar_no = int(input("How many independent variables are you using? "))
ivar_rows = int(input("How many rows are you giving? "))

dvar_value = 0  # Initial value for the dependent variable


# Collect the names of the independent variables
ivar_names = []
for i in range(ivar_no):
    ivar_name = input(f"Enter the name of independent variable {i + 1}: ")
    ivar_names.append(ivar_name)


# Loop to gather values for each row
for j in range(ivar_rows):
    row = {dvar_name: dvar_value}  # Start each row with the dependent variable

    # Collect values for each independent variable in this row
    for ivar_name in ivar_names:
        ivar_value = float(input(f"Enter the value for '{ivar_name}' in row {j + 1}: "))
        row[ivar_name] = ivar_value  # Add each independent variable's value to the row
    if j < ivar_rows:
        row[dvar_name] = int(input(f"The dependent variable of row {j + 1} is:"))

    # Add the completed row to the data list
    data.append(row)

# Create DataFrame directly from the list of dictionaries
df = pd.DataFrame(data)

print(df)

# Save to Excel
df.to_excel(r"C:\Users\ANOINTINGTAMUNOWUNAR\Downloads\baya.xlsx", index=False)
