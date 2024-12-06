import streamlit as st
import pandas as pd

weather = pd.read_csv('pages/weather_raw.csv')
st.table(weather.describe())