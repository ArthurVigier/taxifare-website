import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

st.set_page_config(page_title="TaxiFare + Art Génératif", layout="wide")

st.title("TaxiFare Prediction + Art Génératif")
st.markdown("Prédiction de tarif + visualisation PCA + texte modifié + automates cellulaires animés")

# ───────────────────────────────────────────────
# Colonnes pour les inputs principaux
# ───────────────────────────────────────────────
st.subheader("Paramètres du trajet")

col1, col2 = st.columns([3, 3])
with col1:
    pickup_datetime   = st.text_input("Date et heure prise en charge", "2014-07-06 19:18:00")
    pickup_longitude  = st.number_input("Longitude prise en charge",   value=-73.950655, step=0.0001, format="%.6f")
    pickup_latitude   = st.number_input("Latitude prise en charge",    value=40.783282,  step=0.0001, format="%.6f")
with col2:
    dropoff_longitude = st.number_input("Longitude dépose",  value=-73.984365, step=0.0001, format="%.6f")
    dropoff_latitude  = st.number_input("Latitude dépose",   value=40.769802,  step=0.0001, format="%.6f")
    passenger_count   = st.number_input("Nombre de passagers", value=1, min_value=1, max_value=8, step=1)

# ───────────────────────────────────────────────
# Contrôles artistiques / animation
# ───────────────────────────────────────────────
st.subheader("Contrôles visuels & animation")

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    anim_speed = st.slider("Vitesse animation (s par génération)", 0.05, 0.8, 0.18, step=0.02)
with col_b:
    rule_offset = st.slider("Décalage de la règle Wolfram", 0, 255, 42)
with col_c:
    zoom = st.slider("Taille affichage", 0.6, 3.0, 1.4, step=0.1)
with col_d:
    max_generations = st.slider("Nombre max de générations", 40, 180, 100, step=10)

# ───────────────────────────────────────────────
# Texte à transformer
# ───────────────────────────────────────────────
st.subheader("Texte à transformer (mapping sémantique)")
input_text = st.text_area(
    "Texte original",
    "If you like to gamble I tell you, I'm your man You win some, lose some It's all the same to me",
    height=90
)

# ───────────────────────────────────────────────
# Données de référence pour PCA
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
# Automate cellulaire 1D (Wolfram)
# ───────────────────────────────────────────────
def wolfram_next(state, rule_bin):
    n = len(state)
    new = np.zeros(n, dtype=int)
    for i in range(n):
        left   = state[(i-1) % n]
        center = state[i]
        right  = state[(i+1) % n]
        code   = (left << 2) | (center << 1) | right
        new[i] = int(rule_bin[7 - code])   # règle MSB first
    return new

# ───────────────────────────────────────────────
# BOUTON PRINCIPAL
# ───────────────────────────────────────────────
if st.button("Calculer le prix + Générer l'art + Lancer l'animation", type="primary"):
    with st.spinner("Calcul en cours..."):
        try:
            # ─── Appel API ───────────────────────────────────────
            params = {
                "pickup_datetime": pickup_datetime,
                "pickup_longitude": pickup_longitude,
                "pickup_latitude": pickup_latitude,
                "dropoff_longitude": dropoff_longitude,
                "dropoff_latitude": dropoff_latitude,
                "passenger_count": int(passenger_count),
            }
            r = requests.get("https://taxifare.lewagon.ai/predict", params=params, timeout=8)
            r.raise_for_status()
            fare = r.json().get("fare")
            if fare is None:
                st.error("Impossible de récupérer le prix.")
                st.stop()

            st.success(f"**Prix estimé : ${fare:.2f}**")

            # ─── Features pour seed ──────────────────────────────
            lon1, lat1 = pickup_longitude, pickup_latitude
            lon2, lat2 = dropoff_longitude, dropoff_latitude
            dist_km = ((lon2 - lon1)**2 + (lat2 - lat1)**2)**0.5 * 111
            duration_min = dist_km * 6 + 4

            try:
                dt = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
                hour = dt.hour + dt.minute / 60
                weekend = 1 if dt.weekday() >= 5 else 0
            except:
                hour = 12.0
                weekend = 0

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            seed_base = dist_km + fare + hour + passenger_count * 17 + dlat * 200 + dlon * 300 + weekend * 1000
            np.random.seed(int(seed_base) % 2**32)

            # ─── PCA 2D/3D (affichage rapide) ────────────────────
            pcs, _ = manual_pca(typical_rides)
            current = np.array([dist_km, duration_min, passenger_count, fare, hour, weekend, dlat, dlon])
            curr_pc = (current - np.mean(typical_rides, axis=0)) @ _

            col_left, col_right = st.columns(2)
            with col_left:
                fig2d, ax2d = plt.subplots(figsize=(5, 4))
                ax2d.scatter(pcs[:,0], pcs[:,1], c="lightgray", label="trajets typiques")
                ax2d.scatter(curr_pc[0], curr_pc[1], c="crimson", s=200, marker="*", label="ce trajet")
                ax2d.legend()
                ax2d.set_title("Vue 2D PCA")
                st.pyplot(fig2d)

            with col_right:
                fig3d = plt.figure(figsize=(5, 4))
                ax3d = fig3d.add_subplot(111, projection='3d')
                ax3d.scatter(pcs[:,0], pcs[:,1], pcs[:,2], c="lightgray")
                ax3d.scatter(*curr_pc[:3], c="red", s=150, marker="*")
                ax3d.set_title("Vue 3D sémantique")
                st.pyplot(fig3d)

            # ─── Transformation texte ────────────────────────────
            st.subheader("Texte transformé")
            shift = int(fare * 3 + dist_km * 5) % 26
            transformed = ''.join(
                chr((ord(c) - 97 + shift) % 26 + 97) if c.islower() else c
                for c in input_text.lower()
            )
            st.code(transformed, language=None)

            # ─── Animation Automate Cellulaire ───────────────────
            st.subheader("Animation – Automate cellulaire 1D")

            WIDTH = 64
            initial_state = np.random.randint(0, 2, WIDTH)

            rule_number = (int(fare * 11 + dist_km * 7 + hour * 13 + rule_offset) % 256)
            rule_binary = f"{rule_number:08b}"

            st.markdown(f"Règle Wolfram active : **{rule_number}**  (décalage {rule_offset})")

            placeholder_plot = st.empty()
            current = initial_state.copy()

            for generation in range(max_generations):
                current = wolfram_next(current, rule_binary)

                with placeholder_plot.container():
                    fig, ax = plt.subplots(figsize=(12 * zoom, 3 * zoom))
                    ax.imshow(current.reshape(1, -1), cmap="binary", aspect="auto")
                    ax.set_title(f"Génération {generation+1} / {max_generations}  —  Règle {rule_number}")
                    ax.axis("off")
                    st.pyplot(fig)
                    st.caption(f"État à la génération {generation+1}")

                time.sleep(anim_speed)

            st.success("Animation terminée ! Relancez pour une nouvelle évolution.")

        except Exception as e:
            st.error(f"Erreur rencontrée :\n{str(e)}")

st.caption("Application expérimentale – Prédiction + art procédural via paramètres du trajet • 2025–2026")
