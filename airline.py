import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load("project org.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Airline Satisfaction Predictor", layout="centered")
st.title("✈ Airline Passenger Satisfaction Predictor")

# -----------------------------
# 1️⃣ Human-readable user inputs
# -----------------------------
gender = st.selectbox("Gender", ["Female", "Male"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.selectbox("Class", ["Eco", "Business", "Eco Plus"])
flight_distance = st.number_input("Flight Distance", min_value=1, max_value=5000, value=500)

# Service ratings
wifi = st.slider("Inflight wifi service", 0, 5, 3)
convenient = st.slider("Departure/Arrival time convenient", 0, 5, 3)
online_booking = st.slider("Ease of Online booking", 0, 5, 3)
gate_location = st.slider("Gate location", 0, 5, 3)
food_drink = st.slider("Food and drink", 0, 5, 3)
online_boarding = st.slider("Online boarding", 0, 5, 3)
seat_comfort = st.slider("Seat comfort", 0, 5, 3)
entertainment = st.slider("Inflight entertainment", 0, 5, 3)
onboard_service = st.slider("On-board service", 0, 5, 3)
leg_room = st.slider("Leg room service", 0, 5, 3)
baggage = st.slider("Baggage handling", 0, 5, 3)
checkin = st.slider("Checkin service", 0, 5, 3)
inflight_service = st.slider("Inflight service", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)

# Delays
dep_delay = st.number_input("Departure Delay (Minutes)", min_value=0, max_value=2000, value=0)
arr_delay = st.number_input("Arrival Delay (Minutes)", min_value=0, max_value=2000, value=0)

# -----------------------------
# 2️⃣ Manual mapping for categorical inputs
# -----------------------------
gender_map = {"Female": 0, "Male": 1}
customer_map = {"disloyal Customer": 0, "Loyal Customer": 1}
travel_map = {"Personal Travel": 0, "Business travel": 1}
class_map = {"Eco": 0, "Business": 1, "Eco Plus": 2}

gender_val = gender_map[gender]
customer_val = customer_map[customer_type]
travel_val = travel_map[type_of_travel]
class_val = class_map[travel_class]

# -----------------------------
# 3️⃣ Prediction
# -----------------------------
if st.button("Predict"):
    # Prepare input DataFrame
    input_df = pd.DataFrame([{
        'Gender': gender_val,
        'Customer Type': customer_val,
        'Age': age,
        'Type of Travel': travel_val,
        'Class': class_val,
        'Flight Distance': flight_distance,
        'Inflight wifi service': wifi,
        'Departure/Arrival time convenient': convenient,
        'Ease of Online booking': online_booking,
        'Gate location': gate_location,
        'Food and drink': food_drink,
        'Online boarding': online_boarding,
        'Seat comfort': seat_comfort,
        'Inflight entertainment': entertainment,
        'On-board service': onboard_service,
        'Leg room service': leg_room,
        'Baggage handling': baggage,
        'Checkin service': checkin,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
        'Departure Delay in Minutes': dep_delay,
        'Arrival Delay in Minutes': arr_delay
    }])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Map prediction to human-readable label
    label_map = {0: "Neutral or Dissatisfied", 1: "Satisfied"}
    label = label_map[prediction]



    # Show prediction
    if prediction == 1:
        st.success(f"🟢 Passenger is likely **Satisfied** ({label})")
    else:
        st.error(f"🔴 Passenger is likely **Neutral or Dissatisfied** ({label})")
