import pandas as pd

# Load the Excel file
df = pd.read_excel('C:\\Users\\shien\\Python\\Datasets\\Online Retail.xlsx')

# Drop rows with any missing values
df = df.dropna()

# Function to escape single quotes in text
def escape_quotes(value):
    # Check if the value is a string before attempting to replace quotes
    if isinstance(value, str):
        return value.replace("'", "''")
    return value

# Ensure the InvoiceDate is in datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Format date as string
df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%d/%m/%Y %I:%M:%S %p')

# Filters out rows with negative values for UnitPrice and Quantity
sales_data = df[(df['UnitPrice'] > 0) & (df['Quantity'] > 0)]

# Apply escape_quotes to relevant string columns
sales_data['Description'] = sales_data['Description'].apply(escape_quotes)

# Save the cleaned DataFrame to a new Excel file
try:
    sales_data.to_excel('C:\\Users\\shien\\Python\\Datasets\\Online_Retail_Cleaned_Dataset_2.xlsx', index=False)
    print("Data saved successfully!")
except Exception as e:
    print(f"Error saving data: {e}")
