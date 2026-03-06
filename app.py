import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import difflib

st.set_page_config(page_title="TaxiFare + Game of Life + Info Random", layout="wide")

# ───────────────────────────────────────────────
# Textes de base (version normale)
# ───────────────────────────────────────────────
TEXTS_NORMAL = {
    "app_title": "Prédiction Taxi + Game of Life Art + Info Random",
    "app_desc": """
Cette application combine :
• Prédiction du prix d'un trajet taxi (API Le Wagon)
• Visualisation PCA 2D/3D des caractéristiques du trajet
• Transformation stylée du texte
• Animation Game of Life 2D initialisée à partir des paramètres du trajet
• Recherche d'information inspirée des chiffres des paramètres (via conversion → lettres → mot proche → Wikipedia)
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
    "button_random_info": "Get Random Information",
    "pca_2d_title": "PCA 2D",
    "pca_3d_title": "PCA 3D",
    "transformed_title": "Texte transformé",
    "gol_title": "Game of Life 2D – Animation procédurale",
    "fare_success": "**Prix estimé : ${fare:.2f}**",
    "gol_info": "Grille {size} × {size} • Densité initiale ≈ {density:.1%}",
    "gol_gen_caption": "État à la génération {gen}",
    "gol_gen_title": "Génération {gen} / {max_gens}   –   {alive} cellules vivantes",
    "success_end": "Simulation terminée. Relancez pour une nouvelle évolution !",
    "random_info_title": "Information aléatoire issue des paramètres",
    "error_no_fare": "Impossible de lire le prix depuis l'API.",
    "error_generic": "Erreur : {error}",
    "caption": "TaxiFare + Game of Life procédural + recherche inspirée • déterministe via paramètres • 2026",
}

# ───────────────────────────────────────────────
# État de l'application
# ───────────────────────────────────────────────
if "modified_mode" not in st.session_state:
    st.session_state.modified_mode = False

if "current_texts" not in st.session_state:
    st.session_state.current_texts = TEXTS_NORMAL.copy()

# ───────────────────────────────────────────────
# Fonction de transformation du texte
# ───────────────────────────────────────────────
def transform_text(text: str) -> str:
    if not text:
        return text

    shift = (len(text) % 26) + 1
    result = []
    for idx, c in enumerate(text.lower()):
        if c.isalpha():
            base = ord('a')
            shifted = chr((ord(c) - base + shift) % 26 + base)
            if (idx + shift) % 7 == 0:
                num = (ord(shifted) - ord('a') + 1) % 10
                result.append(str(num))
            else:
                result.append(shifted)
        else:
            result.append(c)

    s = "".join(result).upper()

    flip_freq = 5 + (len(s) % 4)
    parts = [s[i:i+flip_freq] for i in range(0, len(s), flip_freq)]
    for i in range(len(parts)):
        if i % 2 == 1:
            parts[i] = parts[i][::-1]
    return "".join(parts)

# ───────────────────────────────────────────────
# Boutons de contrôle texte (toujours en haut)
# ───────────────────────────────────────────────
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    if st.button("Take & Modify Text Full", type="primary", key="modify_full"):
        st.session_state.modified_mode = True
        for key, val in TEXTS_NORMAL.items():
            st.session_state.current_texts[key] = transform_text(val)
        st.rerun()

with col_btn2:
    if st.button("Go back to normal", key="reset_normal"):
        st.session_state.modified_mode = False
        st.session_state.current_texts = TEXTS_NORMAL.copy()
        st.rerun()

# ───────────────────────────────────────────────
# NLTK ou fallback dictionnaire
# ───────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import words
    # Si besoin : nltk.download('words')  # à faire une seule fois dans votre environnement
    ENGLISH_WORDS = set(w.lower() for w in words.words())
except:
    ENGLISH_WORDS = {
        "taxi", "newyork", "manhattan", "airport", "fare", "traffic", "city", "night",
        "passenger", "route", "latitude", "longitude", "time", "date", "weekend", "hour",
        "distance", "duration", "price", "estimate", "python", "streamlit", "api",
        "wikipedia", "random", "info", "letter", "number", "convert", "match", "hello",
        "world", "test", "data", "code", "app", "button", "click", "fun", "art"
    }

# ───────────────────────────────────────────────
# Fonction conversion chiffres → lettres
# ───────────────────────────────────────────────
def numbers_to_letters(numbers):
    letters = ""
    for num in numbers:
        str_num = str(abs(num)).replace(".", "").replace("-", "")
        for digit in str_num:
            if digit.isdigit():
                letters += chr(ord('A') + int(digit))
    return letters.lower()

# ───────────────────────────────────────────────
# Interface principale
# ───────────────────────────────────────────────
if st.session_state.modified_mode:
    st.title(st.session_state.current_texts["app_title"])
    st.markdown(st.session_state.current_texts["app_desc"])
else:
    st.title(TEXTS_NORMAL["app_title"])
    st.markdown(TEXTS_NORMAL["app_desc"])

if st.session_state.modified_mode:
    st.subheader(st.session_state.current_texts["section_trajet"])
else:
    st.subheader(TEXTS_NORMAL["section_trajet"])

col1, col2 = st.columns([4, 4])
with col1:
    pickup_datetime = st.text_input(
        st.session_state.current_texts["pickup_datetime_label"] if st.session_state.modified_mode else TEXTS_NORMAL["pickup_datetime_label"],
        "2014-07-06 19:18:00"
    )
    pickup_longitude = st.number_input(
        st.session_state.current_texts["pickup_lon_label"] if st.session_state.modified_mode else TEXTS_NORMAL["pickup_lon_label"],
        value=-73.950655, step=0.0001, format="%.6f"
    )
    pickup_latitude = st.number_input(
        st.session_state.current_texts["pickup_lat_label"] if st.session_state.modified_mode else TEXTS_NORMAL["pickup_lat_label"],
        value=40.783282, step=0.0001, format="%.6f"
    )
with col2:
    dropoff_longitude = st.number_input(
        st.session_state.current_texts["dropoff_lon_label"] if st.session_state.modified_mode else TEXTS_NORMAL["dropoff_lon_label"],
        value=-73.984365, step=0.0001, format="%.6f"
    )
    dropoff_latitude = st.number_input(
        st.session_state.current_texts["dropoff_lat_label"] if st.session_state.modified_mode else TEXTS_NORMAL["dropoff_lat_label"],
        value=40.769802, step=0.0001, format="%.6f"
    )
    passenger_count = st.number_input(
        st.session_state.current_texts["passengers_label"] if st.session_state.modified_mode else TEXTS_NORMAL["passengers_label"],
        value=1, min_value=1, max_value=8, step=1
    )

if st.session_state.modified_mode:
    st.subheader(st.session_state.current_texts["section_controls"])
else:
    st.subheader(TEXTS_NORMAL["section_controls"])

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    anim_speed = st.slider(
        st.session_state.current_texts["anim_speed_label"] if st.session_state.modified_mode else TEXTS_NORMAL["anim_speed_label"],
        0.04, 0.9, 0.16, step=0.02
    )
with col_b:
    grid_size = st.slider(
        st.session_state.current_texts["grid_size_label"] if st.session_state.modified_mode else TEXTS_NORMAL["grid_size_label"],
        48, 96, 72, step=4
    )
with col_c:
    zoom_factor = st.slider(
        st.session_state.current_texts["zoom_label"] if st.session_state.modified_mode else TEXTS_NORMAL["zoom_label"],
        0.7, 2.8, 1.45, step=0.1
    )
with col_d:
    max_gens = st.slider(
        st.session_state.current_texts["gens_label"] if st.session_state.modified_mode else TEXTS_NORMAL["gens_label"],
        50, 220, 120, step=10
    )

if st.session_state.modified_mode:
    st.subheader(st.session_state.current_texts["section_text"])
else:
    st.subheader(TEXTS_NORMAL["section_text"])

input_text = st.text_area(
    st.session_state.current_texts["section_text"] if st.session_state.modified_mode else TEXTS_NORMAL["section_text"],
    st.session_state.current_texts["text_placeholder"] if st.session_state.modified_mode else TEXTS_NORMAL["text_placeholder"],
    height=100
)

if st.button(
    st.session_state.current_texts["button_predict"] if st.session_state.modified_mode else TEXTS_NORMAL["button_predict"],
    type="primary"
):
    with st.spinner("Calcul en cours..."):
        try:
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
                st.error(st.session_state.current_texts["error_no_fare"] if st.session_state.modified_mode else TEXTS_NORMAL["error_no_fare"])
                st.stop()

            st.success(
                (st.session_state.current_texts["fare_success"] if st.session_state.modified_mode else TEXTS_NORMAL["fare_success"])
                .format(fare=fare)
            )

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

            seed_value = int(dist_km * 17 + fare * 31 + hour_frac * 13 + passenger_count * 101 + d_lat*999 + d_lon*777)
            np.random.seed(seed_value % 2**32)

            # PCA
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

            X_c = typical_rides - np.mean(typical_rides, axis=0)
            cov = np.cov(X_c.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            idx = np.argsort(eigvals.real)[::-1]
            comp = eigvecs[:, idx[:3]].real
            pcs = X_c @ comp

            current_features = np.array([dist_km, duration_min, passenger_count, fare, hour_frac, is_weekend, d_lat, d_lon])
            curr_proj = (current_features - np.mean(typical_rides, axis=0)) @ comp

            col_pca1, col_pca2 = st.columns(2)
            with col_pca1:
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                ax.scatter(pcs[:,0], pcs[:,1], c="lightgray", label="typique")
                ax.scatter(curr_proj[0], curr_proj[1], c="crimson", s=220, marker="*")
                ax.legend()
                ax.set_title(st.session_state.current_texts["pca_2d_title"] if st.session_state.modified_mode else TEXTS_NORMAL["pca_2d_title"])
                st.pyplot(fig)

            with col_pca2:
                fig3 = plt.figure(figsize=(5.5, 4.5))
                ax3 = fig3.add_subplot(111, projection='3d')
                ax3.scatter(pcs[:,0], pcs[:,1], pcs[:,2], c="lightgray")
                ax3.scatter(*curr_proj[:3], c="red", s=180, marker="*")
                ax3.set_title(st.session_state.current_texts["pca_3d_title"] if st.session_state.modified_mode else TEXTS_NORMAL["pca_3d_title"])
                st.pyplot(fig3)

            # Texte transformé
            if st.session_state.modified_mode:
                st.subheader(st.session_state.current_texts["transformed_title"])
            else:
                st.subheader(TEXTS_NORMAL["transformed_title"])

            transformed = transform_text(input_text)
            st.code(transformed)

            # Game of Life
            if st.session_state.modified_mode:
                st.subheader(st.session_state.current_texts["gol_title"])
            else:
                st.subheader(TEXTS_NORMAL["gol_title"])

            density = 0.14 + (fare / 120) * 0.15 + (dist_km / 60) * 0.09 + (passenger_count / 8) * 0.06
            density = np.clip(density, 0.08, 0.48)

            grid = (np.random.rand(grid_size, grid_size) < density).astype(int)

            if passenger_count >= 4 or is_weekend:
                cx = grid_size // 2
                glider = np.array([[0,1,0],[0,0,1],[1,1,1]])
                grid[cx:cx+3, cx-5:cx-2] = glider

            if st.session_state.modified_mode:
                st.markdown(st.session_state.current_texts["gol_info"].format(size=grid_size, density=density))
            else:
                st.markdown(TEXTS_NORMAL["gol_info"].format(size=grid_size, density=density))

            placeholder = st.empty()
            current = grid.copy()

            for generation in range(1, max_gens + 1):
                current = ((count_neighbors(current) == 3) | ((current == 1) & (count_neighbors(current) == 2))).astype(int)
                alive = np.sum(current)

                with placeholder.container():
                    fig, ax = plt.subplots(figsize=(10 * zoom_factor, 10 * zoom_factor))
                    ax.imshow(current, cmap="binary", interpolation="nearest")
                    title_text = st.session_state.current_texts["gol_gen_title"] if st.session_state.modified_mode else TEXTS_NORMAL["gol_gen_title"]
                    ax.set_title(title_text.format(gen=generation, max_gens=max_gens, alive=alive))
                    ax.axis("off")
                    st.pyplot(fig)

                    caption_text = st.session_state.current_texts["gol_gen_caption"] if st.session_state.modified_mode else TEXTS_NORMAL["gol_gen_caption"]
                    st.caption(caption_text.format(gen=generation))

                time.sleep(anim_speed)

            st.success(st.session_state.current_texts["success_end"] if st.session_state.modified_mode else TEXTS_NORMAL["success_end"])

        except Exception as e:
            error_msg = st.session_state.current_texts["error_generic"] if st.session_state.modified_mode else TEXTS_NORMAL["error_generic"]
            st.error(error_msg.format(error=str(e)))

# ───────────────────────────────────────────────
# Bouton Get Random Information
# ───────────────────────────────────────────────
if st.button(
    st.session_state.current_texts["button_random_info"] if st.session_state.modified_mode else TEXTS_NORMAL["button_random_info"]
):
    with st.spinner("Recherche d'une information inspirée des paramètres..."):
        numbers = [
            pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude,
            passenger_count
        ]
        if 'fare' in locals():
            numbers.append(fare)
        if 'dist_km' in locals():
            numbers.append(dist_km)

        letter_string = numbers_to_letters(numbers)
        st.write(f"**Séquence de lettres dérivée des chiffres** : **{letter_string}** ({len(letter_string)} lettres)")

        query_word = "taxi"  # valeur par défaut absolue

        if ENGLISH_WORDS:
            close_matches = difflib.get_close_matches(
                letter_string,
                ENGLISH_WORDS,
                n=3,
                cutoff=0.35
            )

            if close_matches:
                query_word = close_matches[0]
                st.success(f"Mot(s) le(s) plus proche(s) trouvé(s) : **{', '.join(close_matches)}**")
            else:
                candidates = []
                for word in ENGLISH_WORDS:
                    if len(word) < 4:
                        continue
                    if difflib.SequenceMatcher(None, letter_string[:8], word[:8]).ratio() > 0.22:
                        candidates.append((word, len(word)))

                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    query_word = candidates[0][0]
                    st.info(f"Aucun match fort → mot retenu par sous-chaîne : **{query_word}**")
                else:
                    st.warning("Aucune correspondance même approximative → sujet par défaut : **taxi**")

        st.markdown(f"**Mot finalement utilisé pour la recherche** : **{query_word}**")

        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query_word}"
        try:
            resp = requests.get(wiki_url, timeout=7)
            if resp.status_code == 200:
                data = resp.json()
                if 'extract' in data and data['extract']:
                    summary = data['extract']
                    title = data.get('title', query_word)
                    if st.session_state.modified_mode:
                        st.subheader(f"{st.session_state.current_texts['random_info_title']} – {title}")
                    else:
                        st.subheader(f"{TEXTS_NORMAL['random_info_title']} – {title}")
                    st.markdown(summary[:900] + "..." if len(summary) > 900 else summary)
                    if 'thumbnail' in data and 'source' in data['thumbnail']:
                        st.image(data['thumbnail']['source'], width=280, caption="Illustration Wikipedia")
                else:
                    st.info(f"La page '{query_word}' existe mais n'a pas de résumé court.")
            else:
                st.info(f"Aucune page Wikipedia trouvée pour '{query_word}' (code {resp.status_code}).")
        except Exception as e:
            st.error(f"Problème lors de la requête Wikipedia : {str(e)}")
            st.info(f"Mot recherché : **{query_word}** (vous pouvez essayer manuellement sur wikipedia.org)")

# Caption final
if st.session_state.modified_mode:
    st.caption(st.session_state.current_texts["caption"])
else:
    st.caption(TEXTS_NORMAL["caption"])
