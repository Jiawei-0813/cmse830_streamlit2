import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", 
               page_title="🔧 Behind the Scenes: Data Prep"
               )

st.title("🔧 Behind the Scenes: Data Prep")
st.markdown("""
This page is dedicated to the data preparation process.
We’ll examine raw datasets, clean and merge them, and create new features for deeper insights.
""")

st.divider()

# --- Load Data ---
bike_0 = pd.read_csv('data/bike10pct_raw.csv')
weather_0 = pd.read_csv('data/weather_raw.csv')

# --- Modular Function ---
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

def summarize_var(df):
    """Summarize the dataset variables."""
    numerical_vars = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_vars = df.select_dtypes(include=["object", "category"]).columns
    return numerical_vars, categorical_vars

def check_outliers(df, column):
    """Check for outliers in a numerical column using IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    summary = {
        "outliers": {
            column: {
                "outliers_count": len(outliers),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
        }
    }
    return summary

def display_features(features):
    """Display dataset feature descriptions."""
    for feature in features:
        st.markdown(f"- {feature}")

tabs = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset", "Combined Dataset"])
tab1, tab2, tab3 = tabs

# --- Bike Data ---
with tab1:
    st.header("🚴 Explore London Bike-Sharing Data")

    # Raw Bike Data
    st.subheader("📋 Original Bike Data")
    st.dataframe(bike_0.head())
    st.markdown(f"The original dataset contains **{bike_0.shape[0]:,}** rows and **{bike_0.shape[1]}** columns 
                with both categorical and numerical features.")
        
    st.subheader("Check Data Quality")
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
    
    numeric_bike, categorical_bike = summarize_var(bike_0)
    st.subheader("Summary Statistics")
    tab1_1, tab1_2 = st.columns(2)
    with tab1_1:
        st.subheader("Numerical Variables")

        for num_var in numeric_bike:
            st.write(f"**{num_var}**:")
            st.write(f"Min: {bike_0[num_var].min()}, Max: {bike_0[num_var].max()}, Mean: {bike_0[num_var].mean():.2f}")

            check_outliers(bike_0, num_var)

    with tab1_2:
        st.subheader("Categorical Variables")
        for cat_var in categorical_bike:
            unique_vals = bike_0[cat_var].nunique()
            st.write(f"**{cat_var}**: {unique_vals} unique values")

# --- Weather Data ---
with tab2:
    st.header("🌤️ Explore London Weather Data")

    # Raw Weather Data
    st.subheader("📋 Original Weather Data")
    st.dataframe(weather_0.head())
    st.markdown(f"The original dataset contains **{weather_0.shape[0]:,}** rows and **{weather_0.shape[1]}** 
                columns with all numerical features.")

    st.subheader("Check Data Quality")
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

# --- Combined Data ---
with tab3:
    st.header("🚴‍♂️🌤️ Explore Combined Dataset")
    st.markdown("Combining the cleaned bike-sharing and weather datasets to create a holistic view. "
                "This integration includes time-related and weather-based features, "
                "setting the stage for exploratory data analysis (EDA) and predictive modeling.")


# Footer
st.sidebar.markdown("**Explore London's Bike-Weather Story!**")