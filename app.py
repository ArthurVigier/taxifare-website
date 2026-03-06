import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

st.title("TaxiFare + PCA + Text + Cellular Automata + Animation")

# ───────────────────────────────────────────────
# Inputs de base (comme avant)
# ───────────────────────────────────────────────
st.subheader("Paramètres du trajet")

col1, col2 = st.columns(2)
with col1:
    pickup_datetime   = st.text_input("Pickup datetime",      "2014-07-06 19:18:00")
    pickup_longitude  = st.number_input("Pickup longitude",   value=-73.950655, step=0.0001)
    pickup_latitude   = st.number_input("Pickup latitude",    value=40.783282,  step=0.0001)
with col2:
    dropoff_longitude = st.number_input("Dropoff longitude",  value=-73.984365, step=0.0001)
    dropoff_latitude  = st.number_input("Dropoff latitude",   value=40.769802,  step=0.0001)
    passenger_count   = st.number_input("Passenger count",    value=1, min_value=1, max_value=8, step=1)

# ───────────────────────────────────────────────
# Sliders d’ajustement “artistique” (influence l’initialisation et la règle)
# ───────────────────────────────────────────────
st.subheader("Ajustements visuels / artistiques")

col_a, col_b, col_c = st.columns(3)
with col_a:
    animation_speed = st.slider("Vitesse animation (secondes par génération)", 0.05, 1.0, 0.15, step=0.05)
with col_b:
    rule_offset = st.slider("Décalage règle Wolfram (0–255)", 0, 255, 90, step=1)
with col_c:
    zoom_factor = st.slider("Taille affichage (multiplicateur)", 0.5, 3.0, 1.5, step=0.1)

params_api = {
    "pickup_datetime": pickup_datetime,
    "pickup_longitude": float(pickup_longitude),
    "pickup_latitude": float(pickup_latitude),
    "dropoff_longitude": float(dropoff_longitude),
    "dropoff_latitude": float(dropoff_latitude),
    "passenger_count": int(passenger_count),
}

# ───────────────────────────────────────────────
# Fonctions utiles (automate cellulaire 1D)
# ───────────────────────────────────────────────
def wolfram_step(state, rule_bin):
    n = len(state)
    new_state = np.zeros(n, dtype=int)
    for i in range(n):
        l = state[(i-1) % n]
        c = state[i]
        r = state[(i+1) % n]
        pattern = (l << 2) | (c << 1) | r
        new_state[i] = int(rule_bin[7 - pattern])  # MSB first
    return new_state

def get_rule_bin(rule_num):
    return f"{rule_num:08b}"

# ───────────────────────────────────────────────
# BOUTON PRINCIPAL + ANIMATION
# ───────────────────────────────────────────────
if st.button("Prédire + Générer + Lancer l’animation visuelle", type="primary"):
    with st.spinner("Calcul initial + génération de l’état de départ..."):
        try:
            # ─── Prédiction ─────────────────────────────────────
            resp = requests.get("https://taxifare.lewagon.ai/predict", params=params_api, timeout=8)
            fare = resp.json().get("fare", None)
            if fare is None:
                st.error("Impossible de lire le prix")
                st.stop()

            st.success(f"Prix estimé : **${fare:.2f}**")

            # ─── Features pour seed ─────────────────────────────
            lon1, lat1 = pickup_longitude, pickup_latitude
            lon2, lat2 = dropoff_longitude, dropoff_latitude
            dist = ((lon2-lon1)**2 + (lat2-lat1)**2)**0.5 * 111
            duration = dist * 6 + 4

            try:
                dt = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
                hour = dt.hour + dt.minute/60.0
            except:
                hour = 12.0

            seed_base = dist + fare + hour + passenger_count * 10 + (lat2-lat1)*1000 + (lon2-lon1)*1000
            np.random.seed(int(seed_base) % 2**32)

            # État initial 64 cellules
            WIDTH = 64
            initial = np.random.randint(0, 2, WIDTH)

            # Règle influencée par les sliders + params
            rule = (int(fare * 7 + dist * 11 + hour * 5) + rule_offset) % 256
            rule_bin = get_rule_bin(rule)

            st.markdown(f"**Automate actif** — Règle Wolfram : **{rule}**  |  Vitesse : {animation_speed}s/gén")

            # ─── Zone d’animation ───────────────────────────────
            placeholder = st.empty()
            current_state = initial.copy()

            # On affiche ~80–120 générations (arrêt manuel possible via bouton stop)
            MAX_GEN = 120
            for gen in range(MAX_GEN):
                # Mise à jour
                current_state = wolfram_step(current_state, rule_bin)

                # Affichage
                with placeholder.container():
                    fig, ax = plt.subplots(figsize=(10 * zoom_factor, 2.5 * zoom_factor))
                    ax.imshow(current_state.reshape(1, -1), cmap='binary', aspect='auto')
                    ax.set_title(f"Génération {gen+1}  |  Règle {rule}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    st.pyplot(fig)
                    st.caption(f"État actuel — gen {gen+1}/{MAX_GEN}")

                time.sleep(animation_speed)

                # Sécurité anti-boucle infinie (Streamlit timeout)
                if gen > MAX_GEN - 2:
                    st.info("Animation terminée (limite atteinte). Relancez pour rejouer.")
                    break

        except Exception as e:
            st.error(f"Problème : {str(e)}")

st.caption("Ajustez les sliders → relancez l’animation pour voir l’impact sur la règle et la vitesse. L’état initial reste déterministe (lié aux params du trajet).")
