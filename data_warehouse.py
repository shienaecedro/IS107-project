import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, String, DECIMAL, TIMESTAMP, MetaData
from sqlalchemy.dialects.postgresql import insert

# Database connection
DATABASE_URI = 'postgresql://postgres:admin@localhost:5433/online_retail_store'
engine = create_engine(DATABASE_URI)
metadata = MetaData()

# Define the tables
customers = Table('customers', metadata,
    Column('customer_id', Integer, primary_key=True),
    Column('country', String(50), nullable=False)
)

products = Table('products', metadata,
    Column('stock_code', String(20), primary_key=True),
    Column('description', String(100)),
    Column('unit_price', DECIMAL(10, 2))
)

invoices = Table('invoices', metadata,
    Column('invoice_no', Integer, primary_key=True),
    Column('invoice_date', TIMESTAMP),
    Column('stock_code', String(20)),
    Column('quantity', Integer),
    Column('customer_id', Integer)
)

# Create the tables
metadata.create_all(engine)

# Load the cleaned Excel file
file_path = 'C:\\Users\\shien\\Python\\Datasets\\Online_Retail_Cleaned_Dataset.xlsx'
df = pd.read_excel(file_path)

# Convert 'InvoiceDate' to datetime format, handling the specific format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %I:%M:%S %p')

def insert_customers():
    customers_data = df[['CustomerID', 'Country']].drop_duplicates()
    for _, row in customers_data.iterrows():
        stmt = insert(customers).values(
            customer_id=row['CustomerID'],
            country=row['Country']
        ).on_conflict_do_nothing()
        with engine.begin() as conn:
            conn.execute(stmt)
    print("Customers data inserted successfully!")

def insert_products():
    products_data = df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates()
    for _, row in products_data.iterrows():
        stmt = insert(products).values(
            stock_code=row['StockCode'],
            description=row['Description'],
            unit_price=row['UnitPrice']
        ).on_conflict_do_nothing()
        with engine.begin() as conn:
            conn.execute(stmt)
    print("Products data inserted successfully!")

def insert_invoices():
    invoices_data = df[['InvoiceNo', 'InvoiceDate', 'StockCode', 'Quantity', 'CustomerID']].drop_duplicates()
    for _, row in invoices_data.iterrows():
        stmt = insert(invoices).values(
            invoice_no=row['InvoiceNo'],
            invoice_date=row['InvoiceDate'],
            stock_code=row['StockCode'],
            quantity=row['Quantity'],
            customer_id=row['CustomerID']
        ).on_conflict_do_nothing()
        with engine.begin() as conn:
            conn.execute(stmt)
    print("Invoices data inserted successfully!")

# Call the functions to insert data into the tables
insert_customers()
insert_products()
insert_invoices()
