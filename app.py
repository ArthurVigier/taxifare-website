import streamlit as st
import requests
from datetime import datetime
'''
# TaxiFareModel front
'''

st.markdown('''
Remember that there are several ways to output content into your web page...

Either as with the title by just creating a string (or an f-string). Or as with this paragraph using the `st.` functions
''')

'''

'''

url = 'https://taxifare.lewagon.ai/predict'

#if url == 'https://taxifare.lewagon.ai/predict':
#
#    st.markdown('Maybe you want to use your own API for the prediction, not the one provided by Le Wagon...')

pickup_datetime = st.text_input("Pickup datetime","2014-07-06 19:18:00")
pickup_longitude = st.text_input("Pickup longitude",value=-73.950655)
pickup_latitude = st.text_input("Pickup latitude",value=40.783282)
dropoff_longitude = st.text_input("Dropoff longitude",value=-73.984365)
dropoff_latitude= st.text_input("Dropoff latitude",value=40.769802)
passenger_count = st.text_input("Passenger count",value=1)
#-73.950655
params = {
        'pickup_datetime':pickup_datetime,
        'pickup_longitude':pickup_longitude,
        'pickup_latitude':pickup_latitude,
        'dropoff_longitude':dropoff_longitude,
        'dropoff_latitude':dropoff_latitude,
        'passenger_count':passenger_count,
}

if st.button("PredictFare"):
    response = requests.get(url,params=params)
    prediction = response.json().get("fare")

    st.success(f"Estimated fare ${prediction:.2f}")


'''


'''
