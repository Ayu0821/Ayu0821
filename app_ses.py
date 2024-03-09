import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import streamlit as st

# Read your data
data = pd.read_csv(r'C:\Users\owner\Desktop\project\datasets\Trained data.csv')

# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Month', 'Spice', 'Forecasted Price', 'MAPE', 'Lower Price', 'Upper Price'])

# Get unique spices and states from the DataFrame
spices = data["Spice"].unique()
states = data["State"].unique()

# Streamlit web application
st.title('Price spice forecasting')

# Sidebar for user input
selected_spice = st.sidebar.selectbox('Select Spice', spices)
selected_state = st.sidebar.selectbox('Select State', states)
selected_month = st.sidebar.selectbox('Select Month', range(1, 13))  # Month in number format

# Convert "Month&Year" to datetime format
data['Month&Year'] = pd.to_datetime(data['Month&Year'])

# Filter data based on user input
filtered_data = data[
    (data["Spice"] == selected_spice) &
    (data["State"] == selected_state) &
    (data["Month&Year"].dt.month == selected_month)
]

# Loop through each row in the filtered data
for index, row in filtered_data.iterrows():
    # Get actual prices for the spice
    actual_prices = row["Prices"]

    # Print the data for debugging
    print(f"Row: {row['Month&Year']} - {selected_spice} - {selected_state}")
    print(f"Actual Prices: {actual_prices}")

    # Check if the "Prices" column contains valid numeric data
    if pd.notna(actual_prices) and isinstance(actual_prices, (list, pd.Series, np.ndarray)):
        try:
            # Use Simple Exponential Smoothing (SES) model
            model = SimpleExpSmoothing(actual_prices)
            fit_model = model.fit()
            forecasted_price = fit_model.forecast(1)[0]

            # Calculate MAPE score
            mape_score = mean_absolute_percentage_error(actual_prices, [forecasted_price])

            # Calculate lower and upper prices (optional)
            lower_price = forecasted_price - 0.1 * forecasted_price
            upper_price = forecasted_price + 0.1 * forecasted_price

            # Append the results to the DataFrame
            results_df = results_df.append({
                'Month': selected_month,
                'Spice': selected_spice,
                'Forecasted Price': forecasted_price,
                'MAPE': mape_score,
                'Lower Price': lower_price,
                'Upper Price': upper_price
            }, ignore_index=True)

            print(f"Forecasted Price: {forecasted_price}, MAPE: {mape_score}")
        except Exception as e:
            print(f"Error processing row: {row['Month&Year']} - {selected_spice} - {selected_state}")
            print(f"Error details: {str(e)}")
    else:
        print(f"Invalid data in row: {row['Month&Year']} - {selected_spice} - {selected_state}")

# Display the results in a table
st.table(results_df)





#streamlit run app_ses.py