import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", page_title="About the Data")

## Load Data ##
bike_0 = pd.read_csv('data/bike10pct_raw.csv')
st.table(bike_0.describe())
weather_0 = pd.read_csv('data/weather_raw.csv')
st.table(weather_0.describe())

## Modular Function ##
def check_missing_data(df):
    """Check for missing values and display a summary."""
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing Values": missing_values,
        "% Missing": missing_percent
    })
    if missing_summary["Missing Values"].sum() == 0:
        st.success("No missing values found.")
    else:
        st.dataframe(missing_summary.style.highlight_max(axis=0, color="red").highlight_min(axis=0, color="green"))

def check_duplicates(df):
    """Check for duplicate rows and handle them."""
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        st.warning(f"Number of duplicate rows: {num_duplicates}")
        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.success("Duplicates removed!")
    else:
        st.success("No duplicates found.")
    return df

def display_features(features):
    """Display dataset feature descriptions."""
    for feature in features:
        st.markdown(f"- {feature}")

tabs = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset"])
tab1, tab2 = tabs

#### Bike Data ####
with tab1:
    st.subheader("🚴 London Bike-Sharing Dataset")
    st.markdown("""
                This dataset provides **detailed records** of bike trips managed by the Transport for London (TfL) bike-sharing system.
                """)
    
    st.write("**Data Access:** [Kaggle Dataset](https://www.kaggle.com/datasets/kalacheva/london-bike-share-usage-dataset)")
       
    if st.checkbox("View Raw Bike Data"):
        st.dataframe(bike_0.head())
    
    with st.expander("Check Data Quality"):
        cols1,cols2 = st.column(2)
        with cols1:
            st.write("**Data Types**")
            st.write(bike_0.dtypes.to_frame('Type'))
            check_duplicates(bike_0)

        with cols2: 
            check_duplicates(bike_0)
                    
            st.write("**🛠 Suggested Adjustments**")
            bike_suggestions = [
                "`Start date` and `End date`: Convert to `datetime` format.",
                "`Total duration (ms)`: Convert from milliseconds to minutes.",
                "`Total duration`: Redundant, can be dropped.",
                "`Bike model`: Convert to `category` type."
            ]
            display_features(bike_suggestions)

            st.write("In addition:")
            st.markdown("""
            - Create a **`date`** column in `yyyy-mm-dd HH:MM` based on `Start date` for merging
            - Check consistency between station names and station numbers
            """)

    st.markdown("""
        **Note**: Due to limited storage and the LFS quota being utilized previously, 
        a random 10% of the original data was used for this demonstration.
        """)
    
    st.markdown("**Dataset Coveraage:**")
    bike_coverage = [
        "- **Temporal:** August 1 - August 31, 2023.",
        "- **Spatial:** Bike stations across London.",
        "- **Trips:** 77,653 individual rides (10% of the full dataset)."
    ]
    display_features(bike_coverage)

    st.write('**Features**: ')
    bike_features = ["""
        - **`Number`** (Cardinal): Unique identifier for each trip; Trip ID.  
        - **`Start date`** (Nominal): Date and time in `yyyy-mm-dd hh:mm:ss` format.  
        - **`Start station number`** (Cardinal): Unique identifier for the start station.  
        - **`Start station`** (Nominal, Categorical): Name of the start station.  
        - **`End date`** (Nominal): Date and time in `yyyy-mm-dd hh:mm:ss` format.  
        - **`End station number`** (Cardinal): Unique identifier for the end station.  
        - **`End station`** (Nominal, Categorical): Name of the end station.  
        - **`Bike number`** (Cardinal): Unique identifier for the bike used.  
        - **`Bike model`** (Categorical): Type of bike.  
        - **`Total duration`** (Ratio): Duration of the trip in seconds.  
        - **`Total duration (ms)`** (Ratio): Duration of the trip in milliseconds.  
        """, unsafe_allow_html=True]
    display_features(bike_features)
    
#### Weather Data ####       
with tab2:
    st.subheader('🌤️ London Weather Dataset')
    st.write("""
            The weather dataset contains **historical records of key weather conditions** from Open-Meteo, 
            which partners with national weather services for accuracy. By selecting the most suitable models 
             for each location, Open-Meteo provides **high-resolution weather data**, ensuring reliability for 
             analyzing impacts on activities like bike-sharing. """)
    
    st.subheader("🌤️ London Weather Dataset")
    st.markdown("""
                The weather dataset contains **hourly historical records** of weather conditions provided by Open-Meteo.
                """)
    
    st.write("**Data Access:** [Open-Meteo API Documentation](https://open-meteo.com/en/docs/historical-weather-api)")

    if st.checkbox("View Raw Weather Data"):
        st.dataframe(weather_0.head())

    with st.expander("Check Data Quality"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types:**")
            st.write(weather_0.dtypes.to_frame("Type"))
            check_missing_data(weather_0)
        with col2:
            weather_0 = check_duplicates(weather_0)

    st.markdown("**🛠 Suggested Adjustments:**")
    weather_suggestions = [
        "`date`: Remove timezone information.",
        "`weather_code`: Map numeric codes to descriptions.",
        "`Date`: Extracted from `date` in `yyyy-mm-dd HH:MM` format for merging."
    ]
    display_features(weather_suggestions)

    st.markdown("**Dataset Coverage:**")
    weather_coverage =[
                - **Temporal:** August 1 - August 31, 2023.  
                - **Spatial:** London, United Kingdom (51.5085° N, -0.1257° E).  
    ]
    display_features(weather_coverage)

    st.write("**Features:**")
    weather_features = ["""
        - **`date` (Nominal)**: Date and time of the observation in 'yyyy-mm-dd hh:mm:ss' format.  
        - **`temperature_2m` (Interval)**: Air temperature at 2 meters above ground in degrees Celsius.  
        - **`relative_humidity_2m` (Ratio)**: Relative humidity at 2 meters above ground as a percentage.  
        - **`apparent_temperature` (Interval)**: Feels-like temperature in degrees Celsius, combining wind chill and humidity.  
        - **`wind_speed_10m` (Ratio)**: Wind speed at 10 meters above ground in m/s.  
        - **`wind_direction_10m` (Circular Numeric)**: Wind direction at 10 meters above ground in degrees.  
        - **`weather_code` (Nominal)**: Weather condition represented by a numeric code.  
        """, unsafe_allow_html=True]
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
        st.image('https://github.com/Leftium/weather-sense/assets/381217/6b346c7d-0f10-4976-bb51-e0a8678990b3', use_container_width=True)
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
