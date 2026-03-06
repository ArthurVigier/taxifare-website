import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import difflib
from bs4 import BeautifulSoup  # ← ajouté pour extraire du contenu

st.set_page_config(page_title="TaxiFare + Game of Life", layout="wide")

# ───────────────────────────────────────────────
# Textes de base (version normale) — inchangé
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
    "fare_success": "Prix estimé : ${fare:.2f}",
    "gol_info": "Grille {size} × {size} • Densité initiale ≈ {density:.1%}",
    "gol_gen_caption": "État à la génération {gen}",
    "gol_gen_title": "Génération {gen} / {max_gens}   –   {alive} cellules vivantes",
    "success_end": "Simulation terminée. Relancez pour une nouvelle évolution !",
    "error_no_fare": "Impossible de lire le prix depuis l'API.",
    "error_generic": "Erreur : {error}",
    "caption": "TaxiFare + Game of Life procédural • déterministe via paramètres du trajet • 2026",
    "button_random_info": "Get Random Information",
    "random_info_title": "Information aléatoire issue des params (première page Google organique)",
}

# ───────────────────────────────────────────────
# État et transformation texte (inchangé)
# ───────────────────────────────────────────────
if "modified_mode" not in st.session_state:
    st.session_state.modified_mode = False

if "current_texts" not in st.session_state:
    st.session_state.current_texts = TEXTS_NORMAL.copy()

def transform_text(text: str) -> str:
    if not text.strip():
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

def T(key: str) -> str:
    if not key or key.isspace():
        return key
    return st.session_state.current_texts.get(key, key)
# ───────────────────────────────────────────────
# PCA manuel
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
# Game of Life
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
# Dictionnaire mots pour matching
# ───────────────────────────────────────────────
WORD_DICT = [
    "taxi", "newyork", "manhattan", "airport", "fare", "traffic", "city", "night", "passenger", "route",
    "latitude", "longitude", "time", "date", "weekend", "hour", "distance", "duration", "price", "estimate",
    "python", "streamlit", "api", "wikipedia", "random", "info", "letter", "number", "convert", "match",
    "hello", "world", "test", "data", "code", "app", "button", "click", "fun", "art", "music", "film",
    "science", "history", "space", "earth", "planet", "star", "moon", "sun", "river", "mountain"
]

def numbers_to_letters(numbers):
    letters = ""
    for num in numbers:
        str_num = str(abs(num)).replace(".", "").replace("-", "")
        for digit in str_num:
            if digit.isdigit():
                letters += chr(ord('A') + int(digit))
    return letters.lower()

# ───────────────────────────────────────────────
# Interface
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

# ─── Prédiction + viz ────────────────────────────────
if st.button(T("button_predict"), type="primary"):
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
            r = requests.get("https://taxifare.lewagon.ai/predict", params=params, timeout=9)
            r.raise_for_status()
            fare = r.json().get("fare")
            if fare is None:
                st.error(T("error_no_fare"))
                st.stop()

            st.success(f"**Prix estimé : ${fare:.2f}**")

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
                plt.close(fig)

            with col_pca2:
                fig3 = plt.figure(figsize=(5.5, 4.5))
                ax3 = fig3.add_subplot(111, projection='3d')
                ax3.scatter(pcs[:,0], pcs[:,1], pcs[:,2], c="lightgray")
                ax3.scatter(*curr_proj[:3], c="red", s=180, marker="*")
                ax3.set_title(T("pca_3d_title"))
                st.pyplot(fig3)
                plt.close(fig3)

            st.subheader(T("transformed_title"))
            transformed = transform_text(input_text)
            st.code(transformed)

            st.subheader(T("gol_title"))

            density = 0.14 + (fare / 120) * 0.15 + (dist_km / 60) * 0.09 + (passenger_count / 8) * 0.06
            density = np.clip(density, 0.08, 0.48)

            grid = (np.random.rand(grid_size, grid_size) < density).astype(int)

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
                    plt.close(fig)

                time.sleep(anim_speed)

            st.success(T("success_end"))

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

# ───────────────────────────────────────────────
# Get Random Information
# ───────────────────────────────────────────────
if st.button(T("button_random_info")):
    with st.spinner("Recherche Google en cours..."):
        numbers = [pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count]
        if 'fare' in locals():
            numbers.append(fare)
        if 'dist_km' in locals():
            numbers.append(dist_km)

        letter_string = numbers_to_letters(numbers)
        st.write(f"Chaîne de lettres générée : **{letter_string}**")

        closest = difflib.get_close_matches(letter_string, WORD_DICT, n=1, cutoff=0.2)
        query_word = closest[0] if closest else "taxi"

        st.write(f"Mot le plus proche trouvé : **{query_word}**")

        # Recherche Google (User-Agent pour éviter blocage simple)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        google_url = f"https://www.google.com/search?q={query_word.replace(' ', '+')}"

        try:
            resp = requests.get(google_url, headers=headers, timeout=8)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # On cherche le premier lien organique (pas pub, pas "People also ask", etc.)
            first_link = None
            for g in soup.find_all("div", class_="g"):
                a_tag = g.find("a")
                if a_tag and "href" in a_tag.attrs:
                    href = a_tag["href"]
                    if href.startswith("http") and "google" not in href and "youtube" not in href:
                        first_link = href
                        break

            if not first_link:
                st.warning("Aucun résultat organique clair trouvé sur Google.")
                first_link = "https://www.google.com/search?q=" + query_word.replace(" ", "+")

            st.markdown(f"**Premier résultat organique Google** : [{query_word}]({first_link})")

            # On va chercher ~200 premiers mots du contenu
            try:
                page_resp = requests.get(first_link, headers=headers, timeout=8)
                page_soup = BeautifulSoup(page_resp.text, "html.parser")

                # Extraction naïve : titre + meta description + premiers paragraphes
                title = page_soup.title.string.strip() if page_soup.title else query_word
                meta_desc = page_soup.find("meta", attrs={"name": "description"})
                desc = meta_desc["content"].strip()[:300] if meta_desc else ""

                paragraphs = page_soup.find_all("p")
                text_snippet = ""
                for p in paragraphs:
                    text_snippet += p.get_text(strip=True) + " "
                    if len(text_snippet) > 1200:
                        break

                # Limite ~200 mots
                words = text_snippet.split()
                preview = " ".join(words[:200]) + "..." if len(words) > 200 else text_snippet

                st.subheader(T("random_info_title"))
                st.markdown(f"**{title}**")
                st.markdown(f"_{desc}_" if desc else "")
                st.markdown(preview)
                st.markdown(f"[Lire la page complète →]({first_link})")

            except Exception as e:
                st.info(f"Impossible de lire le contenu de la page ({str(e)}). Lien direct : {first_link}")

        except Exception as e:
            st.error(f"Erreur lors de la recherche Google : {str(e)}")

st.caption(T("caption"))
