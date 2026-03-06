import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.title("TaxiFareModel + 3D Semantic Mapping + Text Modifier")

st.markdown("""
Appel de l’API Taxi Fare + vues 2D/3D PCA-like + **modificateur de texte sémantique** basé sur les params du trajet.
""")

# ───────────────────────────────────────────────
# ──  Inputs trajet (comme avant)  ────────────────
# ───────────────────────────────────────────────
st.subheader("Paramètres du trajet")

col1, col2 = st.columns(2)
with col1:
    pickup_datetime   = st.text_input("Pickup datetime",      "2014-07-06 19:18:00")
    pickup_longitude  = st.number_input("Pickup longitude",   value=-73.950655,  step=0.0001, format="%.6f")
    pickup_latitude   = st.number_input("Pickup latitude",    value=40.783282,   step=0.0001, format="%.6f")
with col2:
    dropoff_longitude = st.number_input("Dropoff longitude",  value=-73.984365,  step=0.0001, format="%.6f")
    dropoff_latitude  = st.number_input("Dropoff latitude",   value=40.769802,   step=0.0001, format="%.6f")
    passenger_count   = st.number_input("Passenger count",    value=1, min_value=1, max_value=8, step=1)

params_api = {
    "pickup_datetime":   pickup_datetime,
    "pickup_longitude":  pickup_longitude,
    "pickup_latitude":   pickup_latitude,
    "dropoff_longitude": dropoff_longitude,
    "dropoff_latitude":  dropoff_latitude,
    "passenger_count":   int(passenger_count),
}

# ───────────────────────────────────────────────
# ──  Nouvelle section : texte à modifier  ────────
# ───────────────────────────────────────────────
st.subheader("Texte à transformer (via mapping sémantique)")
input_text = st.text_area("Entre ton texte ici",
    value="Bonjour je suis un taxi mystique qui va vers l'aéroport avec des passagers joyeux",
    height=120)

# ───────────────────────────────────────────────
# typical_rides (inchangé)
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
    [ 1.2,   6,   1,   8,   13,   0,  0.01,  0.01],
    [ 5.0,  16,   2,  20,   20,   1,  0.03, -0.04],
    [10.0,  30,   4,  38,    6,   0, -0.06,  0.08],
])

def manual_pca(X, n_components=3):
    X_c = X - np.mean(X, axis=0)
    cov = np.cov(X_c.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argsort(eigvals.real)[::-1]
    comp = eigvecs[:, idx[:n_components]].real
    return X_c @ comp, comp

# ───────────────────────────────────────────────
# Bouton principal
# ───────────────────────────────────────────────
if st.button("Prédire + Voir vues + Transformer le texte"):
    with st.spinner("Calcul en cours..."):
        try:
            # Appel API
            resp = requests.get("https://taxifare.lewagon.ai/predict", params=params_api, timeout=6)
            resp.raise_for_status()
            fare = resp.json().get("fare", None)
            if fare is None:
                st.error("Pas de 'fare' dans la réponse API")
                st.stop()

            st.success(f"**Prix estimé : ${fare:.2f}**")

            # ─── Features sémantiques (comme avant) ───────
            lon1, lat1 = pickup_longitude, pickup_latitude
            lon2, lat2 = dropoff_longitude, dropoff_latitude
            dist = np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2) * 111
            duration_min = dist * 6 + 4

            try:
                dt = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
                hour = dt.hour
                is_weekend = 1 if dt.weekday() >= 5 else 0
            except:
                hour, is_weekend = 12, 0

            d_lat = lat2 - lat1
            d_lon = lon2 - lon1

            current = np.array([dist, duration_min, passenger_count, fare, hour, is_weekend, d_lat, d_lon])

            # PCA (juste pour les graphs)
            pcs, components = manual_pca(typical_rides, 3)
            mean = np.mean(typical_rides, axis=0)
            curr_pcs = (current - mean) @ components

            # ─── Affichage 2D & 3D (inchangé, résumé) ─────
            col_fig1, col_fig2 = st.columns(2)
            with col_fig1:
                fig2d, ax = plt.subplots(figsize=(5,4))
                ax.scatter(pcs[:,0], pcs[:,1], c="lightgray", label="typique")
                ax.scatter(curr_pcs[0], curr_pcs[1], c="crimson", marker="*", s=200, label="ce trajet")
                ax.set_title("2D PCA view")
                ax.legend()
                st.pyplot(fig2d)

            with col_fig2:
                fig3d = plt.figure(figsize=(5,4))
                ax3d = fig3d.add_subplot(111, projection='3d')
                ax3d.scatter(pcs[:,0], pcs[:,1], pcs[:,2], c="lightgray")
                ax3d.scatter(*curr_pcs, c="crimson", marker="*", s=200)
                ax3d.set_title("3D Semantic")
                st.pyplot(fig3d)

            # ───────────────────────────────────────────────
            # ──  PARTIE TEXT MODIFIER  ───────────────────────
            # ───────────────────────────────────────────────
            st.subheader("Texte transformé (mapping sémantique)")

            if not input_text.strip():
                st.info("Entre un texte pour voir la transformation.")
            else:
                # Création d'une petite "matrice clé" 3×3 à partir des params
                seed_values = [dist, duration_min, passenger_count, fare, hour, abs(d_lat)+abs(d_lon)]
                seed = sum(seed_values) % 1000
                np.random.seed(int(seed))

                key_matrix = np.random.randn(3,3)
                key_matrix += np.array([[d_lat*10, fare/5, hour/4],
                                        [d_lon*10, -duration_min/3, passenger_count*2],
                                        [is_weekend*5, dist/2, seed/100]])

                key_scalar = np.sum(key_matrix) % 26          # pour César
                key_flip   = int(abs(d_lat*100 + d_lon*100)) % 4  # 0..3 → fréquence inversion

                def modify_char(c, idx):
                    if not c.isalpha():
                        return c

                    # 1. Décalage César
                    shift = int(key_scalar + idx // 5) % 26
                    base = ord('A') if c.isupper() else ord('a')
                    c = chr((ord(c) - base + shift) % 26 + base)

                    # 2. Parfois → chiffre (A→1, B→2...)
                    if (idx + int(fare)) % 7 == 0:
                        num = (ord(c.upper()) - ord('A') + 1) % 10
                        return str(num) if num > 0 else "0"

                    # 3. Inversion segment (tous les key_flip caractères)
                    if idx % (key_flip + 2) == 0 and idx > 3:
                        segment = input_text[max(0,idx-3):idx+1]
                        return segment[::-1][-1]   # dernier du segment inversé

                    return c

                transformed = "".join(modify_char(c, i) for i,c in enumerate(input_text))

                st.markdown("**Texte original** :")
                st.code(input_text)

                st.markdown("**Texte transformé** (selon params du trajet) :")
                st.code(transformed)

                st.caption("Règles : César + substitution lettre→chiffre aléatoire contrôlée + inversions partielles. Déterministe pour mêmes params.")

        except Exception as e:
            st.error(f"Erreur : {e}")

st.caption("Application augmentée — trajet → visualisation + texte modifié sémantiquement")
