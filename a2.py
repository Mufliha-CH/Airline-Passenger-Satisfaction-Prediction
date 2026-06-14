import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(page_title="Flight Satisfaction Predictor", layout="wide")

# --- Load Project Assets ---
@st.cache_resource # Keeps the model in memory for faster performance
def load_assets():
    model = joblib.load("project org.pkl")
    scaler = joblib.load("scaler.pkl")
    le_gender = joblib.load("le.pkl")     # Gender
    le_customer = joblib.load("le1.pkl")   # Customer Type
    le_travel = joblib.load("le2.pkl")     # Type of Travel
    le_class = joblib.load("le3.pkl")      # Class
    return model, scaler, le_gender, le_customer, le_travel, le_class

try:
    model, scaler, le_gender, le_customer, le_travel, le_class = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("✈️ Airline Passenger Satisfaction Predictor")
st.write("Use the sidebar and sliders below to input passenger data and predict their satisfaction level.")

# --- Sidebar: Passenger & Flight Info ---
st.sidebar.header("📋 Passenger Profile")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
age = st.sidebar.number_input("Age", 7, 85, 35)
travel_type = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.sidebar.selectbox("Class", ["Business", "Eco", "Eco Plus"])
flight_dist = st.sidebar.number_input("Flight Distance (miles)", 31, 5000, 800)

# --- Main Page: Service Ratings ---
st.subheader("⭐ Service Ratings (0-5)")
col1, col2, col3 = st.columns(3)

with col1:
    wifi = st.slider("Inflight wifi service", 0, 5, 3)
    time_conv = st.slider("Departure/Arrival time convenient", 0, 5, 3)
    online_book = st.slider("Ease of Online booking", 0, 5, 3)
    gate_loc = st.slider("Gate location", 0, 5, 3)
    food_drink = st.slider("Food and drink", 0, 5, 3)

with col2:
    online_board = st.slider("Online boarding", 0, 5, 3)
    seat_comf = st.slider("Seat comfort", 0, 5, 3)
    entertainment = st.slider("Inflight entertainment", 0, 5, 3)
    onboard_serv = st.slider("On-board service", 0, 5, 3)
    legroom = st.slider("Leg room service", 0, 5, 3)

with col3:
    baggage = st.slider("Baggage handling", 0, 5, 3)
    checkin = st.slider("Checkin service", 0, 5, 3)
    inflight_serv = st.slider("Inflight service", 0, 5, 3)
    cleanliness = st.slider("Cleanliness", 0, 5, 3)

st.subheader("⏱️ Delays")
d_col1, d_col2 = st.columns(2)
with d_col1:
    dep_delay = st.number_input("Departure Delay (mins)", 0, 2000, 0)
with d_col2:
    arr_delay = st.number_input("Arrival Delay (mins)", 0, 2000, 0)

# --- Prediction Logic ---
if st.button("Predict Satisfaction", type="primary"):
    try:
        # 1. Label Encoding (Matching exactly what the LabelEncoders were trained on)
        g_enc = le_gender.transform([gender])[0]
        c_enc = le_customer.transform([customer_type])[0]
        t_enc = le_travel.transform([travel_type])[0]
        cl_enc = le_class.transform([travel_class])[0]

        # 2. Arrange features in the EXACT order (22 columns)
        # Sequence: Gender, Customer Type, Age, Travel Type, Class, Dist, Wifi, ... Clean, Delays
        features = np.array([[
            g_enc, c_enc, age, t_enc, cl_enc, flight_dist,
            wifi, time_conv, online_book, gate_loc, food_drink,
            online_board, seat_comf, entertainment, onboard_serv,
            legroom, baggage, checkin, inflight_serv, cleanliness,
            dep_delay, arr_delay
        ]])

        # 3. Scaling (Crucial for XGBoost performance)
        features_scaled = scaler.transform(features)

        # 4. Prediction
        prediction = model.predict(features_scaled)[0]

        # 5. Display Output
        st.divider()
        if prediction == 1:
            st.success("### Result: The passenger is **SATISFIED** 😊")
            
        else:
            st.warning("### Result: The passenger is **NEUTRAL or DISSATISFIED** 😐")

    except Exception as e:
        st.error(f"Prediction Error: {e}")