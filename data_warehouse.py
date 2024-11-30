import pandas as pd
import psycopg2
from psycopg2 import extras

# Database connection
conn = psycopg2.connect(
    database="online_retail_store",
    user="postgres",
    password="admin",
    host="localhost",
    port="5433"
)

conn.autocommit = True  # Auto-commit each transaction

# Open a cursor to perform database operations
cur = conn.cursor()

# Create the customers table if it doesn't exist
cur.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INT PRIMARY KEY,
        country VARCHAR(50) NOT NULL
    );
""")

# Create the products table if it doesn't exist
cur.execute("""
    CREATE TABLE IF NOT EXISTS products (
        stock_code VARCHAR(20) PRIMARY KEY,
        description VARCHAR(100),
        unit_price DECIMAL(10,2)
    );
""")

# Create the invoices table if it doesn't exist
cur.execute("""
    CREATE TABLE IF NOT EXISTS invoices (
        invoice_no INT,
        invoice_date TIMESTAMP,
        stock_code VARCHAR(20) REFERENCES products(stock_code),
        quantity INT,
        customer_id INT REFERENCES customers(customer_id),
        PRIMARY KEY (invoice_no, stock_code)
    );
""")

# Load the cleaned Excel file
file_path = 'C:\\Users\\shien\\Python\\Datasets\\Online_Retail_Cleaned_Dataset.xlsx'
df = pd.read_excel(file_path)

# Convert 'InvoiceDate' to datetime format, handling the specific format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %I:%M:%S %p')

# Debug: Check the first few rows to ensure dates are correctly parsed
print(df['InvoiceDate'].head())

# Insert Data into Customers table
def insert_customers():
    try:
        customers = df[['CustomerID', 'Country']].drop_duplicates()
        insert_query = """
            INSERT INTO customers (customer_id, country)
            VALUES (%s, %s)
            ON CONFLICT (customer_id) DO NOTHING;
        """
        extras.execute_batch(
            cur, 
            insert_query, 
            customers.to_records(index=False).tolist()
        )
        print("Customers data inserted successfully!")
    except psycopg2.Error as e:
        print(f"Error inserting customers: {e}")

# Insert Data into Products table
def insert_products():
    try:
        products = df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates()
        insert_query = """
            INSERT INTO products (stock_code, description, unit_price)
            VALUES (%s, %s, %s)
            ON CONFLICT (stock_code) DO NOTHING;
        """
        extras.execute_batch(
            cur, 
            insert_query, 
            products.to_records(index=False).tolist()
        )
        print("Products data inserted successfully!")
    except psycopg2.Error as e:
        print(f"Error inserting products: {e}")

# Insert Data into Invoices table
def insert_invoices():
    try:
        invoices = df[['InvoiceNo', 'InvoiceDate', 'StockCode', 'Quantity', 'CustomerID']].drop_duplicates()

        # Convert 'InvoiceDate' to string if necessary
        invoices['InvoiceDate'] = invoices['InvoiceDate'].astype(str)

        insert_query = """
            INSERT INTO invoices (invoice_no, invoice_date, stock_code, quantity, customer_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (invoice_no, stock_code) DO NOTHING;
        """
        extras.execute_batch(
            cur, 
            insert_query, 
            invoices.to_records(index=False).tolist()
        )
        print("Invoices data inserted successfully!")
    except psycopg2.Error as e:
        print(f"Error inserting invoices: {e}")

# Call the functions to insert data into the tables
insert_customers()
insert_products()
insert_invoices()

# Close cursor and connection
cur.close()
conn.close()
