import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

st.set_page_config(page_title="TaxiFare + Game of Life Art", layout="wide")

st.title("Taxi Fare Prediction + Conway's Game of Life Art")
st.markdown("""
Cette application combine :
• Prédiction du prix d'un trajet taxi (API Le Wagon)
• Visualisation PCA 2D/3D des caractéristiques du trajet
• Transformation stylée du texte
• **Animation Game of Life 2D** initialisée à partir des paramètres du trajet
""")

# ───────────────────────────────────────────────
#  Inputs trajet
# ───────────────────────────────────────────────
st.subheader("Paramètres du trajet")

col1, col2 = st.columns([4, 4])
with col1:
    pickup_datetime   = st.text_input("Date et heure",          "2014-07-06 19:18:00")
    pickup_longitude  = st.number_input("Longitude prise en charge", value=-73.950655,  step=0.0001, format="%.6f")
    pickup_latitude   = st.number_input("Latitude prise en charge",  value=40.783282,   step=0.0001, format="%.6f")
with col2:
    dropoff_longitude = st.number_input("Longitude dépose", value=-73.984365,  step=0.0001, format="%.6f")
    dropoff_latitude  = st.number_input("Latitude dépose",  value=40.769802,   step=0.0001, format="%.6f")
    passenger_count   = st.number_input("Passagers",        value=1, min_value=1, max_value=8, step=1)

# ───────────────────────────────────────────────
# Contrôles visuels & animation
# ───────────────────────────────────────────────
st.subheader("Contrôles visuels")

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    anim_speed = st.slider("Vitesse (s par génération)", 0.04, 0.9, 0.16, step=0.02)
with col_b:
    grid_size = st.slider("Taille de la grille", 48, 96, 72, step=4)
with col_c:
    zoom_factor = st.slider("Zoom affichage", 0.7, 2.8, 1.45, step=0.1)
with col_d:
    max_gens = st.slider("Nombre max générations", 50, 220, 120, step=10)

# Texte à transformer
st.subheader("Texte à transformer")
input_text = st.text_area(
    "Texte original",
    "Has he lost his mind? Can he see, or is he blind? Can he walk at all? Or if he moves, will he fall? ",
    height=100
)

# ───────────────────────────────────────────────
# Données typiques pour PCA (features : dist, dur, pass, fare, hour, wknd, dlat, dlon)
# ───────────────────────────────────────────────
typical_rides = np.array([
    [ 1.8,   8,   1,  10,   12,   0,  0.01, -0.02],
    [ 4.2,  14,   1,  18,   18,   0,  0.03, -0.04],
    [ 8.5,  25,   2,  32,    8,   0,  0.05, -0.07],
    [12.0,  35,   1,  45,   22,   1,  0.08, -0.10],
    [20.0,  55,   4,  65,    7,   0,  0.15, -0.18],
    [ 2.5,  11,   3,  14,   14,   0,  0.02, -0.03],
    [ 0.9,   5,   1,   7,   23,   1,  0.005,-0.01],
    [ 3.1,  12,   2,  15,   10,   0, -0.02,  0.03],
    [ 6.4,  20,   1,  25,   16,   0,  0.04,  0.05],
    [15.0,  40,   3,  50,    9,   1, -0.10,  0.12],
])

def manual_pca(X, n_components=3):
    X_c = X - np.mean(X, axis=0)
    cov = np.cov(X_c.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argsort(eigvals.real)[::-1]
    comp = eigvecs[:, idx[:n_components]].real
    return X_c @ comp, comp

# ───────────────────────────────────────────────
# Game of Life functions
# ───────────────────────────────────────────────
def count_neighbors(grid):
    """Compte les voisins (bords toriques)"""
    return (
        np.roll(grid, 1, 0) + np.roll(grid, -1, 0) +
        np.roll(grid, 1, 1) + np.roll(grid, -1, 1) +
        np.roll(np.roll(grid, 1, 0), 1, 1) +
        np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) +
        np.roll(np.roll(grid, -1, 0), -1, 1)
    )

def gol_step(grid):
    neighbors = count_neighbors(grid)
    birth = (neighbors == 3)
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    return np.logical_or(birth, survive).astype(int)

# ───────────────────────────────────────────────
# BOUTON PRINCIPAL
# ───────────────────────────────────────────────
if st.button("Prédire + Visualiser + Animer Game of Life", type="primary"):
    with st.spinner("Calcul en cours..."):
        try:
            # ─── API call ────────────────────────────────────────
            params = {
                "pickup_datetime": pickup_datetime,
                "pickup_longitude": float(pickup_longitude),
                "pickup_latitude": float(pickup_latitude),
                "dropoff_longitude": float(dropoff_longitude),
                "dropoff_latitude": float(dropoff_latitude),
                "passenger_count": int(passenger_count),
            }
            response = requests.get("https://taxifare.lewagon.ai/predict", params=params, timeout=9)
            response.raise_for_status()
            fare = response.json().get("fare")
            if fare is None:
                st.error("Impossible de lire le prix depuis l'API.")
                st.stop()

            st.success(f"**Prix estimé : ${fare:.2f}**")

            # ─── Features pour seed & densité ────────────────────
            lon1, lat1 = pickup_longitude, pickup_latitude
            lon2, lat2 = dropoff_longitude, dropoff_latitude
            dist_km = np.hypot(lon2 - lon1, lat2 - lat1) * 111
            duration_min = dist_km * 6 + 4

            try:
                dt = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
                hour_frac = dt.hour + dt.minute / 60
                is_weekend = 1 if dt.weekday() >= 5 else 0
            except:
                hour_frac = 12.0
                is_weekend = 0

            d_lat = lat2 - lat1
            d_lon = lon2 - lon1

            # Seed global déterministe
            seed_value = int(dist_km * 17 + fare * 31 + hour_frac * 13 + passenger_count * 101 + d_lat*999 + d_lon*777)
            np.random.seed(seed_value % 2**32)

            # ─── PCA ─────────────────────────────────────────────
            pcs, comp = manual_pca(typical_rides)
            current_features = np.array([dist_km, duration_min, passenger_count, fare, hour_frac, is_weekend, d_lat, d_lon])
            curr_proj = (current_features - np.mean(typical_rides, axis=0)) @ comp

            col_pca1, col_pca2 = st.columns(2)
            with col_pca1:
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                ax.scatter(pcs[:,0], pcs[:,1], c="lightgray", label="trajets typiques")
                ax.scatter(curr_proj[0], curr_proj[1], c="crimson", s=220, marker="*", label="ce trajet")
                ax.legend()
                ax.set_title("PCA 2D")
                st.pyplot(fig)

            with col_pca2:
                fig3 = plt.figure(figsize=(5.5, 4.5))
                ax3 = fig3.add_subplot(111, projection='3d')
                ax3.scatter(pcs[:,0], pcs[:,1], pcs[:,2], c="lightgray", s=30)
                ax3.scatter(*curr_proj[:3], c="red", s=180, marker="*")
                ax3.set_title("PCA 3D")
                st.pyplot(fig3)

            # ─── Texte transformé ────────────────────────────────
            st.subheader("Texte transformé")
            shift = int(fare * 5 + dist_km * 7) % 26
            transformed = "".join(
                chr((ord(c) - 97 + shift) % 26 + 97) if c.islower() else c
                for c in input_text.lower()
            )
            st.code(transformed)

            # ─── Game of Life 2D Animation ───────────────────────
            st.subheader("Game of Life 2D – Animation procédurale")

            density = 0.14 + (fare / 120) * 0.15 + (dist_km / 60) * 0.09 + (passenger_count / 8) * 0.06
            density = np.clip(density, 0.08, 0.48)

            grid = (np.random.rand(grid_size, grid_size) < density).astype(int)

            # Petit motif si conditions particulières
            if passenger_count >= 4 or is_weekend:
                cx = grid_size // 2
                glider = np.array([[0,1,0],[0,0,1],[1,1,1]])
                grid[cx:cx+3, cx-5:cx-2] = glider

            st.markdown(f"Grille **{grid_size} × {grid_size}**  •  Densité initiale ≈ **{density:.1%}**")

            placeholder = st.empty()

            current = grid.copy()

            for generation in range(max_gens):
                current = gol_step(current)

                with placeholder.container():
                    fig, ax = plt.subplots(figsize=(10 * zoom_factor, 10 * zoom_factor))
                    ax.imshow(current, cmap="binary", interpolation="nearest")
                    ax.set_title(f"Génération {generation+1} / {max_gens}   –   {np.sum(current)} cellules vivantes")
                    ax.axis("off")
                    st.pyplot(fig)
                    st.caption(f"État à la génération {generation+1}")

                time.sleep(anim_speed)

            st.success("Simulation terminée. Relancez pour une nouvelle évolution !")

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

st.caption("TaxiFare + Game of Life procédural • déterministe via paramètres du trajet • 2025–2026")
