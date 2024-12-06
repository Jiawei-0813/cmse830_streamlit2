import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", 
               page_title="🔧 Behind the Scenes: Data Prep"
               )

st.title("🔧 Behind the Scenes: Data Prep")

st.write("This page is dedicated to the data preparation process. We will load the data, clean it, and prepare it for analysis.")

st.divider()

## Load Data ##
bike_0 = pd.read_csv('data/bike10pct_raw.csv')
weather_0 = pd.read_csv('data/weather_raw.csv')

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

tabs = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset", "Combined Dataset"])
tab1, tab2, tab3 = tabs

#### Bike Data ####
with tab1:
    st.subheader("🚴 London Bike-Sharing Dataset")

    # Raw Bike Data
    st.write("📋 Original Bike Data")
    st.dataframe(bike_0.head())
    st.write(f"The original dataset contains **{bike_0.shape[0]:,}** rows and **{bike_0.shape[1]}** 
             columns with both categorical and numerical features.")
        
    with st.expander("Check Data Quality"):
        cols1,cols2 = st.columns([1,1.5])
        with cols1:
            check_duplicates(bike_0)
            st.write(bike_0.dtypes.to_frame('Data Types'))

        with cols2: 
            check_missing_data(bike_0)
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

#### Weather Data ####
with tab2:
    st.subheader("🌤️ London Weather Dataset")

    # Raw Weather Data
    st.write("📋 Original Weather Data")
    st.dataframe(weather_0.head())
    st.write(f"The original dataset contains **{weather_0.shape[0]:,}** rows and **{weather_0.shape[1]}** 
                columns with all numerical features.")

    with st.expander("Check Data Quality"):
        col1, col2 = st.columns([1,1.6])
        with col1:
            check_duplicates(weather_0)
            st.write(weather_0.dtypes.to_frame("Data Types"))

        with col2:
            check_missing_data(weather_0)
            st.markdown("**🛠 Suggested Adjustments:**")
            weather_suggestions = [
                "`date`: Remove timezone information.",
                "`weather_code`: Map numeric codes to descriptions.",
                "`Date`: Extracted from `date` in `yyyy-mm-dd HH:MM` format for merging."
            ]
            display_features(weather_suggestions)

#### Combined Data ####
with tab3:
    st.subheader("🚴‍♂️🌤️ Combined Dataset")
    st.markdown("Combining the cleaned bike-sharing and weather datasets to create a holistic view. "
                "This integration includes time-related and weather-based features, "
                "setting the stage for exploratory data analysis (EDA) and predictive modeling.")

