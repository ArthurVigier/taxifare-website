import streamlit as st
import requests
import numpy as np
import sounddevice as sd
import time

st.title("Taxi Fare Predictor + Sonification")

st.markdown("""
Enter the ride details below.
When you click **Predict + Sonify**, we'll:
- predict the fare
- map the parameters to sinusoidal frequencies (sonification)
""")

# ────────────────────────────────────────────────
# Inputs
# ────────────────────────────────────────────────
pickup_datetime   = st.text_input("Pickup datetime",   "2014-07-06 19:18:00")
pickup_longitude  = st.number_input("Pickup longitude",  value=-73.950655,  step=0.0001, format="%.6f")
pickup_latitude   = st.number_input("Pickup latitude",   value=40.783282,   step=0.0001, format="%.6f")
dropoff_longitude = st.number_input("Dropoff longitude", value=-73.984365,  step=0.0001, format="%.6f")
dropoff_latitude  = st.number_input("Dropoff latitude",  value=40.769802,   step=0.0001, format="%.6f")
passenger_count   = st.slider("Passenger count", 1, 8, 1)

params = {
    'pickup_datetime': pickup_datetime,
    'pickup_longitude': str(pickup_longitude),
    'pickup_latitude': str(pickup_latitude),
    'dropoff_longitude': str(dropoff_longitude),
    'dropoff_latitude': str(dropoff_latitude),
    'passenger_count': str(passenger_count),
}

# ────────────────────────────────────────────────
# Sonification helper
# ────────────────────────────────────────────────
def sonify_params(params_dict, duration=0.4, sample_rate=44100):
    """
    Very naive but fun sonification:
    - Maps each parameter to a frequency in 200–1800 Hz range
    - Plays short sine waves one after another
    """
    st.write("🎵 Sonifying parameters...")

    # Possible values we want to map (very rough ranges)
    value_ranges = {
        'pickup_longitude':  (-75, -72),
        'pickup_latitude':   (40, 42),
        'dropoff_longitude': (-75, -72),
        'dropoff_latitude':  (40, 42),
        'passenger_count':   (1, 8),
        # datetime we'll skip or hash
    }

    freq_min, freq_max = 220, 1760   # ~A3 → A6

    sounds = []

    for key, value_str in params_dict.items():
        if key == 'pickup_datetime':
            # just skip datetime or use a fixed note / hash
            freq = 440  # A4
        else:
            try:
                val = float(value_str)
                min_v, max_v = value_ranges.get(key, (-100, 100))
                norm = np.clip((val - min_v) / (max_v - min_v + 1e-6), 0, 1)
                freq = freq_min + (freq_max - freq_min) * norm
            except:
                freq = 660  # fallback

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = 0.25 * np.sin(2 * np.pi * freq * t)          # amplitude 0.25
        fade = np.linspace(0, 1, 1000)**1.5                 # very short attack
        fade_out = np.linspace(1, 0, 2000)**2
        tone[:1000] *= fade
        tone[-2000:] *= fade_out
        sounds.append(tone)

    # Concatenate all tones
    full_sound = np.concatenate(sounds)

    # Play
    try:
        sd.play(full_sound, sample_rate)
        sd.wait()
        st.success("Sonification played ✓")
    except Exception as e:
        st.error(f"Could not play sound: {e}\n(is sounddevice working / speakers on?)")

# ────────────────────────────────────────────────
# Prediction + Sonification
# ────────────────────────────────────────────────
col1, col2 = st.columns([1,2])

with col1:
    if st.button("Predict + Sonify", type="primary"):
        with st.spinner("Calling API..."):
            url = "https://taxifare.lewagon.ai/predict"
            try:
                response = requests.get(url, params=params, timeout=8)
                response.raise_for_status()
                data = response.json()
                fare = data.get("fare", None)

                if fare is not None:
                    st.success(f"**Estimated fare: ${fare:.2f}**")
                else:
                    st.error("No 'fare' key in response")
                    st.json(data)

            except Exception as e:
                st.error(f"API call failed\n{str(e)}")

        # Sonify anyway (even if prediction failed)
        sonify_params(params, duration=0.35)

with col2:
    st.caption("What you're hearing (very roughly):")
    st.markdown("""
    • Pickup longitude → lower left = lower pitch
    • Latitudes → higher = higher pitch
    • Passenger count → more people = higher note
    • Pickup time → fixed note
    """)
