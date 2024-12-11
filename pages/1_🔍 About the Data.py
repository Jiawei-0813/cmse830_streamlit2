import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", 
                   page_title="🔍 About the Data")
st.title("🔍 About the Data")

# --- Modular Function ---
def display_features(features):
    """Display dataset feature descriptions."""
    for feature in features:
        st.markdown(f"- {feature}")

tabs = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset"])
tab1, tab2 = tabs

#--- Bike Data ---
with tab1:
    st.subheader("🚴 London Bike-Sharing Dataset")
    st.markdown("""
                The bike dataset provides **detailed records** of bike trips managed by the Transport for London (TfL) bike-sharing system.
                """)
    
    st.write("**Data Access:** [Kaggle Dataset](https://www.kaggle.com/datasets/kalacheva/london-bike-share-usage-dataset)")

    st.markdown("**Dataset Coveraage:**")
    st.markdown("""
        Due to limited storage and the LFS quota being utilized previously, 
        a random 10% of the original data was used for this demonstration.
        """)
    bike_coverage = [
        "**Temporal:** August 1 - August 31, 2023.",
        "**Spatial:** Bike stations across London.",
        "**Trips:** 77,653 individual rides (10% of the full dataset)."
    ]
    display_features(bike_coverage)

    st.write('**Features**: ')
    bike_features = [
        " **`Number`** (Cardinal): Unique identifier for each trip; Trip ID.  ",
        " **`Start date`** (Nominal): Date and time in `yyyy-mm-dd hh:mm:ss` format.",  
        " **`Start station number`** (Cardinal): Unique identifier for the start station.",
        " **`Start station`** (Nominal, Categorical): Name of the start station.  ",
        " **`End date`** (Nominal): Date and time in `yyyy-mm-dd hh:mm:ss` format. ",
        " **`End station number`** (Cardinal): Unique identifier for the end station.  ",
        " **`End station`** (Nominal, Categorical): Name of the end station.  ",
        " **`Bike number`** (Cardinal): Unique identifier for the bike used.  ",
        " **`Bike model`** (Categorical): Type of bike.  ",
        " **`Total duration`** (Ratio): Duration of the trip in seconds.  ",
        " **`Total duration (ms)`** (Ratio): Duration of the trip in milliseconds.  "
        ]
    display_features(bike_features)
    
# --- Weather Data ---  
with tab2:
    st.subheader("🌤️ London Weather Dataset")
    st.write("""
            The weather dataset contains **historical records of key weather conditions** from Open-Meteo, 
             which partners with national weather services for accuracy. By selecting the most suitable models 
             for each location, Open-Meteo provides **high-resolution weather data**, ensuring reliability for 
             analyzing impacts on activities like bike-sharing. """)
    
    st.write("**Data Access:** [Open-Meteo API Documentation](https://open-meteo.com/en/docs/historical-weather-api)")

    with st.expander("Open-Meteo API Code Snippet"):
        st.code("""
        # Install required packages
        # pip install openmeteo-requests
        # pip install requests-cache retry-requests numpy pandas
                
        import openmeteo_requests

        import requests_cache
        import pandas as pd
        from retry_requests import retry

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 51.5085,
            "longitude": -0.1257,
            "start_date": "2023-08-01",
            "end_date": "2023-08-31",
            "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "weather_code", "wind_speed_10m"],
            "timezone": "Europe/London"
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
        hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
        hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
        hourly_data["apparent_temperature"] = hourly_apparent_temperature
        hourly_data["weather_code"] = hourly_weather_code
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        print(hourly_dataframe)

        # Save the data to a CSV file
        hourly_dataframe.to_csv("weather_raw.csv", index = False)
                """)


    st.markdown("**Dataset Coverage:**")
    weather_coverage =[
                "**Temporal:** August 1 - August 31, 2023.",
                "**Spatial:** London, United Kingdom (51.5085° N, -0.1257° E).",
    ]
    display_features(weather_coverage)

    st.write("**Features:**")
    weather_features=[
        " **`date`**(Nominal): Date and time of the observation in 'yyyy-mm-dd hh:mm:ss' format.  ",
        " **`temperature_2m`** (Interval): Air temperature at 2 meters above ground in degrees Celsius. " ,
        " **`relative_humidity_2m`** (Ratio): Relative humidity at 2 meters above ground as a percentage. ",
        " **`apparent_temperature`** (Interval): Feels-like temperature in degrees Celsius, combining wind chill and humidity.  ",
        " **`wind_speed_10m`** (Ratio): Wind speed at 10 meters above ground in m/s.  ",
        " **`wind_direction_10m`** (Circular Numeric): Wind direction at 10 meters above ground in degrees.  ",
        " **`weather_code`** (Nominal): Weather condition represented by a numeric code."  
        ]
    display_features(weather_features)  
    
    # Styling
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] button {
        background-color: transparent;
        color: black;
        font-weight: bold;
    }
    div[data-testid="stSidebar"] button:hover {
        background-color: #90EE90;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hide the weather code information initially
    with st.expander("Show Detailed Weather Code Info"):
        st.image('https://github.com/Leftium/weather-sense/assets/381217/6b346c7d-0f10-4976-bb51-e0a8678990b3', width=800)
        st.write("""
        - **Code Description**:
        - **0**: Clear sky
        - **1, 2, 3**: Mainly clear, partly cloudy, and overcast
        - **45, 48**: Fog and depositing rime fog
        - **51, 53, 55**: Drizzle: Light, moderate, and dense intensity
        - **56, 57**: Freezing Drizzle: Light and dense intensity
        - **61, 63, 65**: Rain: Slight, moderate and heavy intensity
        - **66, 67**: Freezing Rain: Light and heavy intensity
        - **71, 73, 75**: Snow fall: Slight, moderate, and heavy intensity
        - **77**: Snow grains
        - **80, 81, 82**: Rain showers: Slight, moderate, and violent
        - **85, 86**: Snow showers slight and heavy
        - **95***: Thunderstorm: Slight or moderate
        - **96, 99***: Thunderstorm with slight and heavy hail
        - (*) Thunderstorm forecast with hail is only available in Central Europe
        """, unsafe_allow_html=True)

