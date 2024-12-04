import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Set the page configuration to wide mode
st.set_page_config(page_title="Dashboard", layout="wide")

# Database connection function

def get_data(query):
    # Access secrets from the secrets.toml file
    db_host = st.secrets["DB_HOST"]
    db_port = st.secrets["DB_PORT"]
    db_name = st.secrets["DB_NAME"]
    db_user = st.secrets["DB_USER"]
    db_password = st.secrets["DB_PASSWORD"]

    with psycopg.connect(
        f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"
    ) as conn:
        df = pd.read_sql(query, conn)
    return df


### Key Metrics ###

def monthly_sales(start_date_str, end_date_str):
    query_current_month_sales = f"""
    SELECT SUM(p.unit_price * i.quantity) AS monthly_sales 
    FROM invoices i 
    JOIN products p ON i.stock_code = p.stock_code 
    WHERE i.invoice_date BETWEEN '{start_date_str}' AND '{end_date_str}'
    """
    data_current_month_sales = get_data(query_current_month_sales)
    if not data_current_month_sales.empty and data_current_month_sales['monthly_sales'][0] is not None:
        return data_current_month_sales['monthly_sales'][0]
    return 0

def annual_sales(selected_year):
    query_annual_sales = f"""
    SELECT EXTRACT(YEAR FROM i.invoice_date) AS year, 
           SUM(p.unit_price * i.quantity) AS annual_sales 
    FROM invoices i 
    JOIN products p ON i.stock_code = p.stock_code 
    WHERE EXTRACT(YEAR FROM i.invoice_date) = {selected_year}
    GROUP BY EXTRACT(YEAR FROM i.invoice_date)
    ORDER BY EXTRACT(YEAR FROM i.invoice_date);
    """
    data_annual_sales = get_data(query_annual_sales)
    if not data_annual_sales.empty and data_annual_sales['annual_sales'][0] is not None:
        return data_annual_sales['annual_sales'][0]
    return 0

def sales_trend(start_date_str, end_date_str):
    query_sales_trend = f"""
    SELECT i.invoice_date, SUM(p.unit_price * i.quantity) AS sales
    FROM invoices i
    JOIN products p ON i.stock_code = p.stock_code
    WHERE i.invoice_date BETWEEN '{start_date_str}' AND '{end_date_str}'
    GROUP BY i.invoice_date
    ORDER BY i.invoice_date;
    """
    data_sales_trend = get_data(query_sales_trend)
    data_sales_trend['invoice_date'] = pd.to_datetime(data_sales_trend['invoice_date'])
    data_sales_trend['day_of_week'] = data_sales_trend['invoice_date'].dt.day_name()
    
    average_total_sales = data_sales_trend['sales'].mean()

    # Calculate average sales for each day of the week
    day_of_week_sales = data_sales_trend.groupby('day_of_week')['sales'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    # Check if the Series has only NA values
    if day_of_week_sales.dropna().empty:
        highest_sales_day = None
        highest_sales_day_value = None
    else:
        highest_sales_day = day_of_week_sales.idxmax()
        highest_sales_day_value = day_of_week_sales.max()

    # Format the start and end dates
    start_date_fmt = pd.to_datetime(start_date_str).strftime('%b. %d')
    end_date_fmt = pd.to_datetime(end_date_str).strftime('%b. %d')
    year_fmt = pd.to_datetime(start_date_str).strftime('%Y')

    if not data_sales_trend.empty:
        st.line_chart(data_sales_trend.set_index('invoice_date')['sales'])
        
        st.write("**INSIGHTS**")
        st.write(f"The **AVERAGE TOTAL SALES** from {start_date_fmt} to {end_date_fmt} in {year_fmt} is **${average_total_sales:.2f}**.")
        if highest_sales_day:
            st.write(f"**{highest_sales_day}** appears most frequently as the highest sales day with an average sales of **${highest_sales_day_value:.2f}**.")
        else:
            st.write("No valid data for determining the highest sales day.")
    else:
        st.warning("No data available for the selected date range.")

def  top_selling_products(start_date_str, end_date_str):
    query_top_products = f"""
    SELECT p.description, SUM(p.unit_price * i.quantity) AS total_sales
    FROM invoices i
    JOIN products p ON i.stock_code = p.stock_code
    WHERE i.invoice_date BETWEEN '{start_date_str}' AND '{end_date_str}'
    GROUP BY p.description
    ORDER BY total_sales DESC
    LIMIT 10;
    """
    data_top_products = get_data(query_top_products)

    if not data_top_products.empty:
        top_product = data_top_products.iloc[0]['description']
        top_sales = data_top_products.iloc[0]['total_sales']

        seasonal_products = data_top_products[data_top_products['description'].str.contains("CHRISTMAS")]
        seasonal_sales = seasonal_products['total_sales'].sum() if not seasonal_products.empty else 0

        fig_top_products = px.bar(data_top_products, x='description', y='total_sales', labels={'description': 'Products', 'total_sales': 'Total Sales'})
        st.plotly_chart(fig_top_products)

        st.write("**INSIGHTS**")
        st.write(f"""{top_product} significantly outperforms other products with sales reaching **${top_sales:,.2f}**, indicating its popularity and potential for high returns.""")

        if not seasonal_products.empty:
            st.write(f"""Seasonal products like {', '.join(seasonal_products['description'].tolist())} have high sales totaling **${seasonal_sales:,.2f}**, suggesting strong seasonal demand.""")
    else:
        st.warning("No data available for top-selling products.")

### Data Mining ###

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import streamlit as st
from datetime import timedelta

def sales_forecast(start_date_str, end_date_str):
    query_sales = f"""
    SELECT i.invoice_date, SUM(p.unit_price * i.quantity) AS sales
    FROM invoices i
    JOIN products p ON i.stock_code = p.stock_code
    WHERE i.invoice_date BETWEEN '{start_date_str}' AND '{end_date_str}'
    GROUP BY i.invoice_date
    ORDER BY i.invoice_date;
    """
    sales_data = get_data(query_sales)  # Fetch data based on the query
    if sales_data.empty:
        st.warning("No data available for the selected date range.")
        return

    # Convert invoice_date to datetime format
    sales_data['invoice_date'] = pd.to_datetime(sales_data['invoice_date'])

    # Use the ordinal form of the date for the regression
    sales_data['date_ordinal'] = sales_data['invoice_date'].map(pd.Timestamp.toordinal)

    # Define features and target
    X = sales_data[['date_ordinal']]
    y = sales_data['sales']

    if len(X) > 1:
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Build and train the Linear Regression model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
    
        # Make predictions
        y_pred = linear_model.predict(X_test)

        # Create a DataFrame for actual vs predicted sales
        results_df = pd.DataFrame({'Invoice Date': X_test['date_ordinal'].map(pd.Timestamp.fromordinal), 'Actual': y_test, 'Predicted': y_pred})
        results_df = results_df.sort_values(by='Invoice Date')

        # Plot the actual vs predicted sales using Streamlit and Plotly
        fig = px.scatter(results_df, x='Invoice Date', y='Actual', labels={'Invoice Date': 'Date', 'Actual': 'Sales'})
        fig.add_scatter(x=results_df['Invoice Date'], y=results_df['Predicted'], mode='lines', name='Predicted')
        st.plotly_chart(fig)

        # Predict sales for the next month
        next_month_date = pd.to_datetime(end_date_str) + timedelta(days=30)
        future_data = pd.DataFrame({'date_ordinal': [next_month_date.toordinal()]})
    
        # Predict sales for next month
        predicted_sales = linear_model.predict(future_data)

    else:
        st.warning("Not enough data to split into training and test sets.")





def customer_segmentation(start_date_str, end_date_str):
    query_customer = f"""
    SELECT c.customer_id, COUNT(i.invoice_no) AS purchase_frequency, SUM(p.unit_price * i.quantity) AS total_spending
    FROM customers c
    JOIN invoices i ON c.customer_id = i.customer_id
    JOIN products p ON i.stock_code = p.stock_code
    WHERE i.invoice_date BETWEEN '{start_date_str}' AND '{end_date_str}'
    GROUP BY c.customer_id;
    """
    customer_data = get_data(query_customer)

    if not customer_data.empty:
        # Calculate average spending per purchase for each customer
        customer_data['average_spending'] = customer_data['total_spending'] / customer_data['purchase_frequency']

        # Define thresholds for low, medium, and high-value purchases
        low_value_threshold = 50
        high_value_threshold = 200

        # Categorize customers
        customer_data['value_category'] = pd.cut(customer_data['average_spending'],
                                                 bins=[-float('inf'), low_value_threshold, high_value_threshold, float('inf')],
                                                 labels=['Low Value', 'Medium Value', 'High Value'])

        # Count customers in each category
        value_counts = customer_data['value_category'].value_counts()
        low_value_customers_count = value_counts.get('Low Value', 0)
        medium_value_customers_count = value_counts.get('Medium Value', 0)
        high_value_customers_count = value_counts.get('High Value', 0)

        # Preprocess data for clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(customer_data[['purchase_frequency', 'total_spending']])

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(scaled_data)
        customer_data['cluster'] = kmeans.labels_

        # Visualize the clusters
        fig_clusters = px.scatter(customer_data, x='purchase_frequency', y='total_spending', color='cluster', labels={'total_spending': 'Total Spending', 'purchase_frequency': 'Purchase Frequency'})
        st.plotly_chart(fig_clusters)

        # Display insights
        st.write("**INSIGHTS**")
        st.write(f"{low_value_customers_count} customers with spending below ${low_value_threshold} per purchase are categorized as **Low Value Customers** customers.")
        st.write(f"{medium_value_customers_count} customers spending between ${low_value_threshold} and ${high_value_threshold} per purchase are categorized as **Medium Value** customers.")
        st.write(f"{high_value_customers_count} customers spending above ${high_value_threshold} per purchase are categorized as **High Value** customers.")
    else:
        st.warning("No data available for customer segmentation.")
# Main app code
def main():
    st.title('Dashboard')

    # Get the current date
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    # Set default start and end dates to the current month
    default_start_date = datetime(current_year, current_month, 1)
    default_end_date = (default_start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)

    # Date inputs for filtering data
    st.sidebar.subheader('Select Date Range')
    start_date = st.sidebar.date_input('Start Date', value=default_start_date, min_value=datetime(2000, 1, 1))
    end_date = st.sidebar.date_input('End Date', value=default_end_date, min_value=start_date)

    # Year input for filtering annual sales
    st.sidebar.subheader('Select Year')
    selected_year = st.sidebar.selectbox('Year', list(range(2000, current_year + 1)), index=list(range(2000, current_year + 1)).index(current_year))

    if start_date > end_date:
        st.error("Error: End Date must fall after Start Date.")
    else:
        # Convert date inputs to strings
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Format the start and end dates
        start_date_fmt = pd.to_datetime(start_date_str).strftime('%b. %d')
        end_date_fmt = pd.to_datetime(end_date_str).strftime('%b. %d')
        year_fmt = pd.to_datetime(start_date_str).strftime('%Y')

        ### Monthly Sales and Annual Sales ###
        current_month_sales = monthly_sales(start_date_str, end_date_str)
        current_annual_sales = annual_sales(selected_year)

        col1, col2 = st.columns(2)
        with col1:
            st.header('Monthly Sales')
            if current_month_sales:
                st.metric(label="Total", value=f"${current_month_sales:,.2f}")
            else:
                st.warning("No data available for the selected month.")

        with col2:
            st.header(f'Annual Sales ({selected_year})')
            if current_annual_sales:
                st.metric(label="Total", value=f"${current_annual_sales:,.2f}")
            else:
                st.warning("No data available for the selected year.")

        ### Sales Trend and Popular Products ###
        col3, col4 = st.columns(2)
        with col3:
            st.header(f'Sales Trend ({start_date_fmt} to {end_date_fmt} {year_fmt})')
            sales_trend(start_date_str, end_date_str)

        with col4:
            st.header('Sales Forecast')
            sales_forecast(start_date_str, end_date)

        ### Top-Selling Products and Customer Segmentation ###
        col5, col6 = st.columns(2)
        with col5:
            st.header("Top-Selling Products")
            top_selling_products(start_date_str, end_date_str)

        with col6:
            st.header("Customer Segmentation")
            customer_segmentation(start_date_str, end_date_str)

# Call the main function
if __name__ == "__main__":
    main()