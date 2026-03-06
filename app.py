import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.title("TaxiFare + 3D PCA + Text Modifier + Cellular Automata")

st.markdown("""
Taxi Fare API + vues PCA 2D/3D + modificateur texte + **automates cellulaires 1D**
initialisés via matrices 8×8 dérivées des paramètres du trajet.
""")

# ───────────────────────────────────────────────
# Inputs trajet
# ───────────────────────────────────────────────
st.subheader("Paramètres du trajet")

col1, col2 = st.columns(2)
with col1:
    pickup_datetime   = st.text_input("Pickup datetime",      "2014-07-06 19:18:00")
    pickup_longitude  = st.number_input("Pickup longitude",   value=-73.950655, step=0.0001, format="%.6f")
    pickup_latitude   = st.number_input("Pickup latitude",    value=40.783282,  step=0.0001, format="%.6f")
with col2:
    dropoff_longitude = st.number_input("Dropoff longitude",  value=-73.984365, step=0.0001, format="%.6f")
    dropoff_latitude  = st.number_input("Dropoff latitude",   value=40.769802,  step=0.0001, format="%.6f")
    passenger_count   = st.number_input("Passenger count",    value=1, min_value=1, max_value=8, step=1)

params_api = {
    "pickup_datetime":   pickup_datetime,
    "pickup_longitude":  pickup_longitude,
    "pickup_latitude":   pickup_latitude,
    "dropoff_longitude": dropoff_longitude,
    "dropoff_latitude":  dropoff_latitude,
    "passenger_count":   int(passenger_count),
}

# Texte à transformer (comme avant)
st.subheader("Texte à transformer")
input_text = st.text_area("Texte",
    "Le taxi file dans la nuit new-yorkaise vers un destin incertain", height=80)

# ───────────────────────────────────────────────
# typical_rides + PCA manuelle (inchangé)
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
# Fonction automate cellulaire 1D (règle de Wolfram)
# ───────────────────────────────────────────────
def wolfram_ca_1d(initial_state, rule, steps):
    """
    initial_state : array 1D de 0/1 (longueur 64 ou autre)
    rule : entier 0–255 (règle Wolfram)
    steps : nb de générations
    """
    n = len(initial_state)
    history = np.zeros((steps, n), dtype=int)
    history[0] = initial_state

    rule_bin = f"{rule:08b}"[::-1]  # LSB à MSB

    for t in range(1, steps):
        for i in range(n):
            left   = history[t-1, (i-1) % n]
            center = history[t-1, i]
            right  = history[t-1, (i+1) % n]
            pattern = (left << 2) | (center << 1) | right
            history[t, i] = int(rule_bin[pattern])

    return history

# ───────────────────────────────────────────────
# Bouton
# ───────────────────────────────────────────────
if st.button("Tout lancer : Prédire + Vues + Text + Automates"):
    with st.spinner("Calcul..."):
        try:
            # API
            resp = requests.get("https://taxifare.lewagon.ai/predict", params=params_api, timeout=8)
            resp.raise_for_status()
            fare = resp.json().get("fare", None)
            if fare is None:
                st.error("Pas de fare")
                st.stop()

            st.success(f"Prix estimé : **${fare:.2f}**")

            # Features
            lon1, lat1 = pickup_longitude, pickup_latitude
            lon2, lat2 = dropoff_longitude, dropoff_latitude
            dist_km = np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2) * 111
            duration_min = dist_km * 6 + 4

            try:
                dt = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
                hour = dt.hour + dt.minute/60
                is_weekend = 1 if dt.weekday() >= 5 else 0
            except:
                hour, is_weekend = 12.0, 0

            d_lat = lat2 - lat1
            d_lon = lon2 - lon1

            current = np.array([dist_km, duration_min, passenger_count, fare, hour, is_weekend, d_lat, d_lon])

            # PCA (pour les vues 2D/3D – on garde)
            pcs, comp = manual_pca(typical_rides, 3)
            mean = np.mean(typical_rides, axis=0)
            curr_pcs = (current - mean) @ comp

            colA, colB = st.columns(2)
            with colA:
                fig, ax = plt.subplots(figsize=(5,4))
                ax.imshow(pcs[:,:2], aspect='auto', cmap='gray')
                ax.set_title("PCA points 2D")
                st.pyplot(fig)
            with colB:
                fig3d = plt.figure(figsize=(5,4))
                ax3d = fig3d.add_subplot(111, projection='3d')
                ax3d.scatter(*pcs.T, c='gray', s=20)
                ax3d.scatter(*curr_pcs, c='red', s=120, marker='*')
                st.pyplot(fig3d)

            # ─── Text modifier (simplifié ici pour place) ───
            st.subheader("Texte transformé")
            seed_text = int((dist_km + fare + hour) * 100) % 999999
            np.random.seed(seed_text)
            shift = seed_text % 26
            transformed = ''.join(chr((ord(c) - 97 + shift) % 26 + 97) if c.islower() else c for c in input_text.lower())
            st.code(transformed)

            # ─── AUTOMATES CELLULAIRES ────────────────────────
            st.subheader("Automates cellulaires 1D issus des params")

            # Valeurs pour générer différents seeds / règles
            base_values = [dist_km, duration_min, passenger_count, fare, hour, d_lat*100, d_lon*100, is_weekend*50]
            base_sum = sum(base_values)

            # On va générer 4–6 automates différents
            n_automata = 5
            steps = 60
            width = 64   # 8×8 aplati

            fig_ca, axes = plt.subplots(1, n_automata, figsize=(4*n_automata, 5), sharey=True)

            for i in range(n_automata):
                # Différents offsets pour varier
                offset = i * 17 + int(base_sum * (i+1)) % 10000
                seed_float = sum(v * (i+1.3)**2 for v in base_values) + offset
                np.random.seed(int(seed_float) % 2**32)

                # Création état initial 64 bits
                # Méthode : on prend plusieurs floats → bits via mantisse / hash simple
                floats = np.array([dist_km*3.1, fare*7.7, hour*11.1, d_lat*99, d_lon*101]) + offset/1000
                bits = np.unpackbits(np.array(floats.view(np.uint8)))[-width:]

                # Règle Wolfram dérivée des params
                rule = int((fare*13 + dist_km*7 + hour*5 + passenger_count*101 + offset) % 256)

                # Evolution
                evolution = wolfram_ca_1d(bits, rule, steps)

                # Affichage
                ax = axes[i]
                ax.imshow(evolution, cmap='binary', interpolation='nearest')
                ax.set_title(f"Rule {rule}\nseed offset {offset%1000}")
                ax.set_xticks([])
                ax.set_yticks([])

            st.pyplot(fig_ca)

            st.caption("Chaque automate = état initial 64 cellules + règle Wolfram différents → dérivés des params (dist, prix, heure, deltas…). Déterministe.")

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

st.caption("TaxiFare + art génératif via automates cellulaires • 2025–2026 vibe")
