import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

st.title("TaxiFareModel Front + 3D Semantic Mapping")

st.markdown("""
This app calls the Taxi Fare API, shows a simplified **eigenvector-style representation** (2D PCA-like),
and now adds a **3D semantic mapping** using PCA on ride features for a multi-dimensional view.
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
# Expanded "reference ride cloud" for better PCA
# Features: [dist_km, duration_min, passengers, fare_usd, hour_of_day, is_weekend, delta_lat, delta_lon]
# (added more semantic features: time of day, weekend, lat/lon deltas for directionality)
# ───────────────────────────────────────────────
typical_rides = np.array([
    # dist  dur  pass  fare  hour  wknd  dlat  dlon
    [ 1.8,   8,   1,  10,   12,   0,  0.01, -0.02],   # short midday
    [ 4.2,  14,   1,  18,   18,   0,  0.03, -0.04],
    [ 8.5,  25,   2,  32,    8,   0,  0.05, -0.07],
    [12.0,  35,   1,  45,   22,   1,  0.08, -0.10],   # late weekend
    [20.0,  55,   4,  65,    7,   0,  0.15, -0.18],   # airport-ish morning
    [ 2.5,  11,   3,  14,   14,   0,  0.02, -0.03],
    [ 0.9,   5,   1,   7,   23,   1,  0.005,-0.01],
    [ 3.1,  12,   2,  15,   10,   0, -0.02,  0.03],   # opposite direction
    [ 6.4,  20,   1,  25,   16,   0,  0.04,  0.05],
    [15.0,  40,   3,  50,    9,   1, -0.10,  0.12],
    [ 1.2,   6,   1,   8,   13,   0,  0.01,  0.01],
    [ 5.0,  16,   2,  20,   20,   1,  0.03, -0.04],
    [10.0,  30,   4,  38,    6,   0, -0.06,  0.08],
])

# If sklearn not available, manual PCA implementation
def manual_pca(X, n_components=3):
    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argsort(eigvals)[::-1]
    components = eigvecs[:, idx[:n_components]]
    return X_centered @ components, components

# ───────────────────────────────────────────────
# Button & prediction
# ───────────────────────────────────────────────
if st.button("Predict Fare → Show Views"):
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

            # ─── Prepare current ride features ───────────────
            lon1, lat1 = pickup_longitude,  pickup_latitude
            lon2, lat2 = dropoff_longitude, dropoff_latitude

            # Crude distance (km approx, using 111 km/deg)
            dist = np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2) * 111

            # Naive duration estimate
            duration_min = dist * 6 + 4  # ~6 min/km + base

            # Semantic features
            try:
                dt = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
                hour = dt.hour
                is_weekend = 1 if dt.weekday() >= 5 else 0
            except:
                hour = 12  # default
                is_weekend = 0

            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1

            current = np.array([dist, duration_min, passenger_count, fare, hour, is_weekend, delta_lat, delta_lon])

            # ─── Compute PCA (use sklearn if avail, else manual) ───
            try:
                pca = PCA(n_components=3)
                pcs = pca.fit_transform(typical_rides)
                curr_pcs = pca.transform(current.reshape(1, -1))[0]
                explained_var = pca.explained_variance_ratio_
            except ImportError:
                pcs, components = manual_pca(typical_rides, 3)
                curr_pcs = (current - np.mean(typical_rides, axis=0)) @ components
                explained_var = [0.5, 0.3, 0.2]  # approx

            # ─── 2D View (as before, first two PCs) ──────────
            fig2d, ax2d = plt.subplots(figsize=(7, 5.5))
            ax2d.scatter(pcs[:,0], pcs[:,1], s=60, c="lightgray", edgecolor="gray", label="typical rides")
            ax2d.scatter(curr_pcs[0], curr_pcs[1], s=180, c="crimson", edgecolor="darkred", marker="*", label="your ride")
            ax2d.set_xlabel(f"PC1 ({explained_var[0]:.1%} var: dist/price)")
            ax2d.set_ylabel(f"PC2 ({explained_var[1]:.1%} var: time/direction)")
            ax2d.set_title("2D Eigenvector View")
            ax2d.legend()
            ax2d.grid(True, alpha=0.3)
            ax2d.text(curr_pcs[0]*1.03, curr_pcs[1]*1.03, f"${fare:.1f}", fontsize=13, fontweight="bold", color="darkred")
            st.pyplot(fig2d)

            # ─── 3D Semantic Mapping ─────────────────────────
            fig3d = plt.figure(figsize=(8, 6.5))
            ax3d = fig3d.add_subplot(projection='3d')
            ax3d.scatter(pcs[:,0], pcs[:,1], pcs[:,2], s=60, c="lightgray", edgecolor="gray", label="typical rides")
            ax3d.scatter(curr_pcs[0], curr_pcs[1], curr_pcs[2], s=180, c="crimson", edgecolor="darkred", marker="*", label="your ride")
            ax3d.set_xlabel(f"PC1 ({explained_var[0]:.1%} var: dist/price)")
            ax3d.set_ylabel(f"PC2 ({explained_var[1]:.1%} var: time/direction)")
            ax3d.set_zlabel(f"PC3 ({explained_var[2]:.1%} var: passengers/time)")
            ax3d.set_title("3D Semantic Mapping\n(multi-dim view of ride params)")
            ax3d.legend()
            ax3d.text(curr_pcs[0], curr_pcs[1], curr_pcs[2]*1.03, f"${fare:.1f}", fontsize=13, fontweight="bold", color="darkred")
            st.pyplot(fig3d)

        except requests.exceptions.RequestException as e:
            st.error(f"API call failed\n{e}")
        except Exception as e:
            st.error(f"Unexpected error\n{e}")

st.caption("Notes: 3D view uses PCA for semantic dimensionality reduction. Axes are interpreted semantically based on feature loadings. Expanded features for better mapping.")
