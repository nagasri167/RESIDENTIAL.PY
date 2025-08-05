import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Title and setup
st.set_page_config(page_title="🔌 Energy Usage Predictor", layout="centered")
st.title("🔌 Residential Energy Usage Predictor")
st.markdown("Predict appliance energy usage (kWh) for given days of the year.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your energy usage CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    
    # Feature engineering
    df["Days"] = df["Date"].dt.day_name()
    df["dates"] = df["Date"].dt.dayofyear
    X = df[["dates"]]
    
    try:
        y = df["Appliance_Usage_kWh"]
    except KeyError:
        st.error("Column 'Appliance_Usage_kWh' not found in uploaded file.")
        st.stop()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Select prediction range
    st.sidebar.subheader("📅 Prediction Days")
    start_day = st.sidebar.number_input("Start Day (1-365)", min_value=1, max_value=365, value=214)
    end_day = st.sidebar.number_input("End Day (1-365)", min_value=1, max_value=365, value=220)

    if start_day > end_day:
        st.error("Start day cannot be after end day.")
    else:
        # Create prediction input
        predict_df = pd.DataFrame({"dates": list(range(start_day, end_day + 1))})
        prediction = model.predict(predict_df)

        # Display predictions
        result_df = predict_df.copy()
        result_df["Predicted_kWh"] = prediction
        st.subheader("🔍 Predictions")
        st.dataframe(result_df)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(result_df["dates"], result_df["Predicted_kWh"], marker='o', color='teal')
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Predicted Appliance Usage (kWh)")
        ax.set_title("Predicted Energy Usage")
        ax.grid(True)
        st.pyplot(fig)
