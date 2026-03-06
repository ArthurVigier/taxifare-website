import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.title("TaxiFareModel Front + Eigenvector-style view")

st.markdown("""
This app calls the Taxi Fare API and shows a simplified **eigenvector-style representation**
(very approximate 2D PCA-like projection) of inputs and predicted fare.
""")

# ───────────────────────────────────────────────
# Input fields
# ───────────────────────────────────────────────
pickup_datetime   = st.text_input("Pickup datetime",      "2014-07-06 19:18:00")
pickup_longitude  = st.number_input("Pickup longitude",   value=-73.950655,  step=0.0001, format="%.6f")
pickup_latitude   = st.number_input("Pickup latitude",    value=40.783282,   step=0.0001, format="%.6f")
dropoff_longitude = st.number_input("Dropoff longitude",  value=-73.984365,  step=0.0001, format="%.6f")
dropoff_latitude  = st.number_input("Dropoff latitude",   value=40.769802,   step=0.0001, format="%.6f")
passenger_count   = st.number_input("Passenger count",    value=1,           min_value=1, max_value=8, step=1)

# Prepare parameters for API
params = {
    "pickup_datetime":   pickup_datetime,
    "pickup_longitude":  pickup_longitude,
    "pickup_latitude":   pickup_latitude,
    "dropoff_longitude": dropoff_longitude,
    "dropoff_latitude":  dropoff_latitude,
    "passenger_count":   int(passenger_count),
}

# ───────────────────────────────────────────────
# Very rough "reference ride cloud" for visualization
# (these are approximate centers / scales — in real life you would compute real PCA)
# ───────────────────────────────────────────────
#               dist   duration(min)  pass  fare($)
typical_rides = np.array([
    [ 1.8,   8,   1,  10],   # short Manhattan
    [ 4.2,  14,   1,  18],
    [ 8.5,  25,   2,  32],
    [12.0,  35,   1,  45],
    [20.0,  55,   4,  65],   # airport-ish
    [ 2.5,  11,   3,  14],
    [ 0.9,   5,   1,   7],
])

mean = typical_rides.mean(axis=0)
std  = typical_rides.std(axis=0) + 1e-6

# Two strongest "directions" (very hand-crafted approximation)
v1 = np.array([0.65, 0.55, 0.15, 0.48])   # mostly distance + fare
v2 = np.array([-0.38, 0.62, -0.68, 0.12]) # duration vs passengers

# ───────────────────────────────────────────────
# Button & prediction
# ───────────────────────────────────────────────
if st.button("Predict Fare → Show Eigenview"):
    with st.spinner("Calling API..."):
        try:
            response = requests.get("https://taxifare.lewagon.ai/predict", params=params, timeout=6)
            response.raise_for_status()
            data = response.json()
            fare = data.get("fare", None)

            if fare is None:
                st.error("Could not read 'fare' from API response")
                st.json(data)
                st.stop()

            st.success(f"**Estimated fare: ${fare:.2f}**")

            # ─── Prepare current point ────────────────────────────────
            # Very rough feature engineering (same as typical_rides)
            lon1, lat1 = pickup_longitude,  pickup_latitude
            lon2, lat2 = dropoff_longitude, dropoff_latitude

            # crude euclidean distance in degrees (~ rough)
            dist = np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2) * 111

            # very naive duration estimate
            duration_min = dist * 6 + 4               # ~6 min/km + base

            current = np.array([dist, duration_min, passenger_count, fare])

            # Project everything
            X_centered = (typical_rides - mean) / std
            current_c  = (current     - mean) / std

            pc1 = X_centered @ v1
            pc2 = X_centered @ v2

            curr_pc1 = current_c @ v1
            curr_pc2 = current_c @ v2

            # ─── Plot ─────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(7, 5.5))

            ax.scatter(pc1, pc2, s=60, c="lightgray", edgecolor="gray", label="typical rides")
            ax.scatter(curr_pc1, curr_pc2, s=180, c="crimson", edgecolor="darkred",
                       marker="*", label="your ride")

            ax.set_xlabel("≈ PC1  (distance + price)")
            ax.set_ylabel("≈ PC2  (time ↔ passengers)")
            ax.set_title("Simplified Eigenvector View\n(your ride vs typical rides)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # annotate current point
            ax.text(curr_pc1*1.03, curr_pc2*1.03,
                    f"${fare:.1f}", fontsize=13, fontweight="bold", color="darkred")

            st.pyplot(fig)

        except requests.exceptions.RequestException as e:
            st.error(f"API call failed\n{e}")
        except Exception as e:
            st.error(f"Unexpected error\n{e}")

st.caption("Note: this is a **very simplified** 2D projection — not real PCA. Just for fun & intuition.")
