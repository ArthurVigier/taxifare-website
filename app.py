import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

st.set_page_config(page_title="TaxiFare + Game of Life", layout="wide")

# ───────────────────────────────────────────────
# Textes de base (version normale)
# ───────────────────────────────────────────────
TEXTS_NORMAL = {
    "app_title": "Prédiction Taxi + Game of Life Art",
    "app_desc": """
Cette application combine :
• Prédiction du prix d'un trajet taxi (API Le Wagon)
• Visualisation PCA 2D/3D des caractéristiques du trajet
• Transformation stylée du texte
• **Animation Game of Life 2D** initialisée à partir des paramètres du trajet
""",
    "section_trajet": "Paramètres du trajet",
    "pickup_datetime_label": "Date et heure prise en charge",
    "pickup_lon_label": "Longitude prise en charge",
    "pickup_lat_label": "Latitude prise en charge",
    "dropoff_lon_label": "Longitude dépose",
    "dropoff_lat_label": "Latitude dépose",
    "passengers_label": "Nombre de passagers",
    "section_controls": "Contrôles visuels",
    "anim_speed_label": "Vitesse (s par génération)",
    "grid_size_label": "Taille de la grille",
    "zoom_label": "Zoom affichage",
    "gens_label": "Nombre max générations",
    "section_text": "Texte à transformer",
    "text_placeholder": "Un taxi traverse la nuit new-yorkaise, entre lumières et ombres, vers un destin incertain...",
    "button_predict": "Prédire + Visualiser + Animer Game of Life",
    "button_modify_full": "Take & Modify Text Full",
    "button_reset": "Go back to normal",
    "pca_2d_title": "PCA 2D",
    "pca_3d_title": "PCA 3D",
    "transformed_title": "Texte transformé",
    "gol_title": "Game of Life 2D – Animation procédurale",
    "fare_success": "**Prix estimé : ${fare:.2f}**",
    "gol_info": "Grille {size} × {size} • Densité initiale ≈ {density:.1%}",
    "gol_gen_caption": "État à la génération {gen}",
    "gol_gen_title": "Génération {gen} / {max_gens}   –   {alive} cellules vivantes",
    "success_end": "Simulation terminée. Relancez pour une nouvelle évolution !",
    "error_no_fare": "Impossible de lire le prix depuis l'API.",
    "error_generic": "Erreur : {error}",
    "caption": "TaxiFare + Game of Life procédural • déterministe via paramètres du trajet • 2025–2026",
}

# ───────────────────────────────────────────────
# État de l'application
# ───────────────────────────────────────────────
if "modified_mode" not in st.session_state:
    st.session_state.modified_mode = False

if "current_texts" not in st.session_state:
    st.session_state.current_texts = TEXTS_NORMAL.copy()

# ───────────────────────────────────────────────
# Fonction de transformation du texte (basée sur les règles précédentes)
# ───────────────────────────────────────────────
def transform_text(text: str) -> str:
    if not text:
        return text

    # Shift César basé sur longueur
    shift = (len(text) % 26) + 1
    result = []
    for idx, c in enumerate(text.lower()):
        if c.isalpha():
            # César
            base = ord('a')
            shifted = chr((ord(c) - base + shift) % 26 + base)
            # Substitution lettre → chiffre si condition
            if (idx + shift) % 7 == 0:
                num = (ord(shifted) - ord('a') + 1) % 10
                result.append(str(num))
            else:
                result.append(shifted)
        else:
            result.append(c)

    s = "".join(result).upper()

    # Inversion partielle de segments (tous les 5-8 chars)
    flip_freq = 5 + (len(s) % 4)
    parts = [s[i:i+flip_freq] for i in range(0, len(s), flip_freq)]
    for i in range(len(parts)):
        if i % 2 == 1:
            parts[i] = parts[i][::-1]
    return "".join(parts)

# ───────────────────────────────────────────────
# Boutons de contrôle (toujours visibles)
# ───────────────────────────────────────────────
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    if st.button(TEXTS_NORMAL["button_modify_full"], type="primary", key="modify_full"):
        st.session_state.modified_mode = True
        for key, val in TEXTS_NORMAL.items():
            st.session_state.current_texts[key] = transform_text(val)
        st.rerun()

with col_btn2:
    if st.button(TEXTS_NORMAL["button_reset"], key="reset_normal"):
        st.session_state.modified_mode = False
        st.session_state.current_texts = TEXTS_NORMAL.copy()
        st.rerun()

st.markdown(
    """
    <small style="color: #777; font-style: italic;">
    • « Take & Modify Text Full » → remplace tous les titres, labels, descriptions et boutons par leur version transformée (César + substitutions + inversions)<br>
    • « Go back to normal » → rétablit immédiatement l'interface en français standard
    </small>
    """,
    unsafe_allow_html=True
)
# ───────────────────────────────────────────────
# Raccourci pour texte actuel
# ───────────────────────────────────────────────
def T(key: str) -> str:
    return st.session_state.current_texts.get(key, key)

# ───────────────────────────────────────────────
# Données typiques pour PCA
# ───────────────────────────────────────────────
typical_rides = np.array([
    [1.8, 8, 1, 10, 12, 0, 0.01, -0.02],
    [4.2, 14, 1, 18, 18, 0, 0.03, -0.04],
    [8.5, 25, 2, 32, 8, 0, 0.05, -0.07],
    [12.0, 35, 1, 45, 22, 1, 0.08, -0.10],
    [20.0, 55, 4, 65, 7, 0, 0.15, -0.18],
    [2.5, 11, 3, 14, 14, 0, 0.02, -0.03],
    [0.9, 5, 1, 7, 23, 1, 0.005, -0.01],
    [3.1, 12, 2, 15, 10, 0, -0.02, 0.03],
    [6.4, 20, 1, 25, 16, 0, 0.04, 0.05],
    [15.0, 40, 3, 50, 9, 1, -0.10, 0.12],
])

def manual_pca(X, n_components=3):
    X_c = X - np.mean(X, axis=0)
    cov = np.cov(X_c.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argsort(eigvals.real)[::-1]
    comp = eigvecs[:, idx[:n_components]].real
    return X_c @ comp, comp

# ───────────────────────────────────────────────
# Fonctions Game of Life
# ───────────────────────────────────────────────
def count_neighbors(grid):
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
    return ((neighbors == 3) | ((grid == 1) & (neighbors == 2))).astype(int)

# ───────────────────────────────────────────────
# Interface principale
# ───────────────────────────────────────────────
st.title(T("app_title"))
st.markdown(T("app_desc"))

st.subheader(T("section_trajet"))

col1, col2 = st.columns([4, 4])
with col1:
    pickup_datetime = st.text_input(T("pickup_datetime_label"), "2014-07-06 19:18:00")
    pickup_longitude = st.number_input(T("pickup_lon_label"), value=-73.950655, step=0.0001, format="%.6f")
    pickup_latitude = st.number_input(T("pickup_lat_label"), value=40.783282, step=0.0001, format="%.6f")
with col2:
    dropoff_longitude = st.number_input(T("dropoff_lon_label"), value=-73.984365, step=0.0001, format="%.6f")
    dropoff_latitude = st.number_input(T("dropoff_lat_label"), value=40.769802, step=0.0001, format="%.6f")
    passenger_count = st.number_input(T("passengers_label"), value=1, min_value=1, max_value=8, step=1)

st.subheader(T("section_controls"))

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    anim_speed = st.slider(T("anim_speed_label"), 0.04, 0.9, 0.16, step=0.02)
with col_b:
    grid_size = st.slider(T("grid_size_label"), 48, 96, 72, step=4)
with col_c:
    zoom_factor = st.slider(T("zoom_label"), 0.7, 2.8, 1.45, step=0.1)
with col_d:
    max_gens = st.slider(T("gens_label"), 50, 220, 120, step=10)

st.subheader(T("section_text"))
input_text = st.text_area(T("section_text"), T("text_placeholder"), height=100)

if st.button(T("button_predict"), type="primary"):
    with st.spinner("Calcul en cours..."):
        try:
            # ─── API ─────────────────────────────────────────────
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
                st.error(T("error_no_fare"))
                st.stop()

            st.success(T("fare_success").format(fare=fare))

            # ─── Features ────────────────────────────────────────
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

            # Seed déterministe
            seed_value = int(dist_km * 17 + fare * 31 + hour_frac * 13 + passenger_count * 101 + d_lat*999 + d_lon*777)
            np.random.seed(seed_value % 2**32)

            # ─── PCA ─────────────────────────────────────────────
            pcs, comp = manual_pca(typical_rides)
            current_features = np.array([dist_km, duration_min, passenger_count, fare, hour_frac, is_weekend, d_lat, d_lon])
            curr_proj = (current_features - np.mean(typical_rides, axis=0)) @ comp

            col_pca1, col_pca2 = st.columns(2)
            with col_pca1:
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                ax.scatter(pcs[:,0], pcs[:,1], c="lightgray", label="typique")
                ax.scatter(curr_proj[0], curr_proj[1], c="crimson", s=220, marker="*")
                ax.legend()
                ax.set_title(T("pca_2d_title"))
                st.pyplot(fig)

            with col_pca2:
                fig3 = plt.figure(figsize=(5.5, 4.5))
                ax3 = fig3.add_subplot(111, projection='3d')
                ax3.scatter(pcs[:,0], pcs[:,1], pcs[:,2], c="lightgray")
                ax3.scatter(*curr_proj[:3], c="red", s=180, marker="*")
                ax3.set_title(T("pca_3d_title"))
                st.pyplot(fig3)

            # ─── Texte transformé ────────────────────────────────
            st.subheader(T("transformed_title"))
            transformed = transform_text(input_text)
            st.code(transformed)

            # ─── Game of Life ────────────────────────────────────
            st.subheader(T("gol_title"))

            density = 0.14 + (fare / 120) * 0.15 + (dist_km / 60) * 0.09 + (passenger_count / 8) * 0.06
            density = np.clip(density, 0.08, 0.48)

            grid = (np.random.rand(grid_size, grid_size) < density).astype(int)

            # Ajout motif si conditions
            if passenger_count >= 4 or is_weekend:
                cx = grid_size // 2
                glider = np.array([[0,1,0],[0,0,1],[1,1,1]])
                grid[cx:cx+3, cx-5:cx-2] = glider

            st.markdown(T("gol_info").format(size=grid_size, density=density))

            placeholder = st.empty()
            current = grid.copy()

            for generation in range(1, max_gens + 1):
                current = gol_step(current)
                alive = np.sum(current)

                with placeholder.container():
                    fig, ax = plt.subplots(figsize=(10 * zoom_factor, 10 * zoom_factor))
                    ax.imshow(current, cmap="binary", interpolation="nearest")
                    ax.set_title(T("gol_gen_title").format(gen=generation, max_gens=max_gens, alive=alive))
                    ax.axis("off")
                    st.pyplot(fig)
                    st.caption(T("gol_gen_caption").format(gen=generation))

                time.sleep(anim_speed)

            st.success(T("success_end"))

        except Exception as e:
            st.error(T("error_generic").format(error=str(e)))

st.caption(T("caption"))
# ───────────────────────────────────────────────
# Section : Multi-prédictions aléatoires + Heatmap des variations
# ───────────────────────────────────────────────
st.subheader("Multi-prédictions aléatoires + Heatmap des variations de prix")

with st.expander("Hyperparamètres (contrôles)", expanded=False):
    nb_appels = st.slider("Nombre d'appels API (max 5)", 1, 5, 3)
    variation_lon_lat = st.slider("Variation max longitude/latitude (°)", 0.0, 0.05, 0.015, step=0.001)
    variation_pass = st.slider("Variation max passagers", 0, 3, 1)
    seed_aleatoire = st.number_input("Seed aléatoire (pour reproductibilité)", value=42, step=1)

if st.button("Lancer les prédictions multiples + Heatmap"):

    if nb_appels < 1 or nb_appels > 5:
        st.error("Le nombre d'appels doit être entre 1 et 5.")
    else:
        with st.spinner(f"Appels API en cours ({nb_appels} prédictions)..."):

            np.random.seed(seed_aleatoire)

            results = []
            base_params = {
                "pickup_datetime": pickup_datetime,
                "pickup_longitude": pickup_longitude,
                "pickup_latitude": pickup_latitude,
                "dropoff_longitude": dropoff_longitude,
                "dropoff_latitude": dropoff_latitude,
                "passenger_count": int(passenger_count),
            }

            for i in range(nb_appels):
                # Copie des paramètres de base
                p = base_params.copy()

                # Petites perturbations aléatoires
                p["pickup_longitude"]  += np.random.uniform(-variation_lon_lat, variation_lon_lat)
                p["pickup_latitude"]   += np.random.uniform(-variation_lon_lat, variation_lon_lat)
                p["dropoff_longitude"] += np.random.uniform(-variation_lon_lat, variation_lon_lat)
                p["dropoff_latitude"]  += np.random.uniform(-variation_lon_lat, variation_lon_lat)
                p["passenger_count"]    = max(1, min(8, p["passenger_count"] + np.random.randint(-variation_pass, variation_pass + 1)))

                try:
                    resp = requests.get("https://taxifare.lewagon.ai/predict", params=p, timeout=6)
                    resp.raise_for_status()
                    fare = resp.json().get("fare", None)

                    if fare is not None:
                        results.append({
                            "call": i+1,
                            "fare": fare,
                            "passengers": p["passenger_count"],
                            "pickup_lon": p["pickup_longitude"],
                            "pickup_lat": p["pickup_latitude"],
                            "dropoff_lon": p["dropoff_longitude"],
                            "dropoff_lat": p["dropoff_latitude"],
                        })
                except Exception as e:
                    st.warning(f"Appel {i+1} échoué : {str(e)}")

            if not results:
                st.error("Aucune prédiction valide obtenue.")
            else:
                st.success(f"{len(results)} prédictions réussies sur {nb_appels} tentatives")

                # ─── Création d'une petite matrice pour heatmap ───────
                fares = [r["fare"] for r in results]
                if len(fares) >= 2:
                    # On crée une matrice carrée approximative
                    side = int(np.ceil(np.sqrt(len(fares))))
                    pad_len = side * side - len(fares)
                    padded_fares = fares + [np.nan] * pad_len
                    heatmap_data = np.array(padded_fares).reshape(side, side)

                    # Heatmap simple 2D (plus fiable que 3D)
                    fig, ax = plt.subplots(figsize=(7, 6))
                    im = ax.imshow(heatmap_data, cmap='YlOrRd', interpolation='nearest')
                    ax.set_title(f"Heatmap des prix prédits\n({len(fares)} valeurs – variations aléatoires)")
                    ax.set_xlabel("Index prédiction")
                    ax.set_ylabel("Index prédiction")
                    plt.colorbar(im, ax=ax, label="Prix estimé ($)")

                    # Annotations des valeurs
                    for i in range(side):
                        for j in range(side):
                            val = heatmap_data[i,j]
                            if not np.isnan(val):
                                text = f"{val:.1f}"
                                ax.text(j, i, text, ha="center", va="center",
                                        color="black" if val > np.nanmean(heatmap_data)/2 else "white",
                                        fontsize=9)

                    st.pyplot(fig)

                    # Tableau récapitulatif
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results.round(4), use_container_width=True)

                else:
                    st.info("Pas assez de résultats pour produire une heatmap significative.")
