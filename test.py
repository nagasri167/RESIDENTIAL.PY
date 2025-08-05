import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Constants
DATA_FILE = "residential_energy_usage.csv"

# Load and prepare data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    df["Days"] = df["Date"].dt.day_name()
    return df

# Train the model
def train_model(df):
    df_copy = df.copy()
    df_copy["Datess"] = df_copy["Date"].dt.dayofyear
    x = df_copy["Datess"].values.reshape(-1, 1)
    y = df_copy["Appliance_Usage_kWh"]
    ml = LinearRegression()
    ml.fit(x, y)
    y_pred = ml.predict(x)
    r2 = r2_score(y, y_pred)
    return ml, r2

# Predict future usage
def future_prediction(ml, start_day, start_date):
    future_days = pd.DataFrame({"Datess": list(range(start_day, start_day + 7))})
    predictions = ml.predict(future_days)
    future_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=7)
    fut_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Energy Usage (kWh)": predictions
    })
    return fut_df

# Plot usage trends
def plot_usage(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["Date"], df["Appliance_Usage_kWh"], label="Daily usage")
    ax.plot(df["Date"], df["Appliance_Usage_kWh"].rolling(window=7).mean(), label="7-day Average", linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy Usage (kWh)")
    ax.set_title("Energy Usage Over Time")
    ax.legend()
    st.pyplot(fig)

# Streamlit App Configuration
st.set_page_config(page_title="🏠Residential Energy Consumption ⚡", layout="centered")
st.title("🏠 Residential Energy Consumption Dashboard ⚡")

# Load data
df = load_data()

# Section: Add new entry
st.header("📌 Add Today's Usage")
with st.form("entry_form"):
    new_date = st.date_input("Date")
    new_usage = st.number_input("Energy usage (kWh)", min_value=0.0, step=0.1)
    submit = st.form_submit_button("Add Entry")
    if submit:
        new_row = pd.DataFrame({
            "Date": [pd.to_datetime(new_date)],
            "Appliance_Usage_kWh": [new_usage],
            "Days": [pd.to_datetime(new_date).day_name()]
        })
        df_up = pd.concat([df, new_row], ignore_index=True)
        df_up = df_up.sort_values("Date").drop_duplicates("Date")
        df_up.to_csv(DATA_FILE, index=False)
        st.success("✅ Entry added!")
        st.experimental_rerun()

# Section: Show energy trend
st.header("📊 Energy Usage Trend")
plot_usage(df)

# Section: Show prediction
st.header("🔮 Next 7 Days Prediction")
ml_model, r2 = train_model(df)
start_day = df["Date"].dt.dayofyear.max() + 1
start_date = df["Date"].max()
pred_df = future_prediction(ml_model, start_day, start_date)

st.write(f"**Model R² Score**: {r2:.3f}")
st.dataframe(pred_df, use_container_width=True)

# Optional: Download data
st.header("⬇️ Download Data")
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button("Download Full Dataset as CSV", csv_data, file_name='residential_energy_usage.csv', mime='text/csv')
