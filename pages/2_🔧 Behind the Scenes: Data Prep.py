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

def categorize_var(df):
    """Summarize the dataset variables."""
    numerical_vars = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_vars = df.select_dtypes(include=["object", "category"]).columns
    return numerical_vars, categorical_vars

def check_outliers(df, column):
    """Check for outliers in a numerical column using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

def summarize(df, dataset_name="Dataset"):
    """Display summary statistics for numeric and categorical variables."""
    numeric_vars, categorical_vars = categorize_var(df)

    st.subheader(f"Summary Statistics")
    num_tab, cat_tab = st.tabs(["Numerical Variables", "Categorical Variables"])

    # Numerical Analysis
    with num_tab:
        st.subheader("Numerical Variables")
        if len(numeric_vars) > 0: 
            # Display descriptive statistics
            st.dataframe(df[numeric_vars].describe().transpose().round(2))
            st.subheader("Outlier Detection")
            for num_var in numeric_vars:
                outliers_count, lower, upper = check_outliers(df, num_var)
        else:
            st.write("No numerical variables found in the dataset.")

    # Categorical Analysis
    with cat_tab:
        st.subheader("Categorical Variables")
        if len(categorical_vars) > 0:  
            unique_vals = df[categorical_vars].nunique()
            st.write("Unique Value Counts")
            st.dataframe(unique_vals.to_frame(name="Unique Values"))

            selected_cat_var = st.selectbox(
                "Select a categorical variable to analyze:", categorical_vars
            )
            if selected_cat_var:
                value_counts = df[selected_cat_var].value_counts()
                st.markdown(f"### Value Counts for `{selected_cat_var}`")
                st.bar_chart(value_counts)
        else:
            st.write("No categorical variables found in the dataset.")

def display_features(features):
    """Display dataset feature descriptions."""
    for feature in features:
        st.markdown(f"- {feature}")

# --- Sidebar ---
step = st.sidebar.selectbox("Go to", ["📋 Original Data", "🧹 Data Cleaning", "✅ Cleaned Data"])

# --- Main Content ---
bike_tab, weather_tab, combined_tab = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset", "Combined Dataset"])

# --- Bike Data ---
with bike_tab:
    st.header("🚴 Explore London Bike-Sharing Data")

    if step == "📋 Original Data":
        st.subheader("📋 Original Bike Data")
        st.dataframe(bike_0.head())
        st.markdown(f"""The original dataset contains **{bike_0.shape[0]:,}** rows and **{bike_0.shape[1]}** columns 
                    with both categorical and numerical features.""")
            
        if st.checkbox("Check Data Quality", key="bike_quality"):
            cols1, cols2 = st.columns([1, 1.5])
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
        
    elif step == "🧹 Data Cleaning":
        st.subheader("🧹 Data Cleaning")
        st.markdown("Based on the data quality checks, we will now clean the dataset.")
        st.markdown("1. **Data Type Conversion**")
        st.markdown("2. **Feature Engineering**")
        st.markdown("3. **Handling Missing Data**")
        st.markdown("4. **Dropping Redundant Columns**")
        st.markdown("5. **Consistency Checks**")
        st.markdown("6. **Save the Cleaned Dataset**")

    else:
        # Cleaned Bike Data
        st.subheader("✅ Cleaned Bike Data")
        summarize(bike_0, "Bike Dataset")

# --- Weather Data ---
with weather_tab:
    st.header("🌤️ Explore London Weather Data")

    if step == "📋 Original Data":
        st.subheader("📋 Original Weather Data")
        st.dataframe(weather_0.head())
        st.markdown(f"""The original dataset contains **{weather_0.shape[0]:,}** rows and **{weather_0.shape[1]}** 
                    columns with all numerical features.""")

        if st.checkbox("Check Data Quality", key="weather_quality"):
            col1, col2 = st.columns([1, 1.6])
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
        
    elif step == "🧹 Data Cleaning":
        st.subheader("🧹 Data Cleaning")
        st.markdown("Based on the data quality checks, we will now clean the dataset.")
        st.markdown("1. **Data Type Conversion**")
        st.markdown("2. **Feature Engineering**")
        st.markdown("3. **Handling Missing Data**")
        st.markdown("4. **Dropping Redundant Columns**")
        st.markdown("5. **Save the Cleaned Dataset**")

    else:
        # Cleaned Weather Data
        st.subheader("✅ Cleaned Weather Data ")
        summarize(weather_0, "Weather Dataset")

# --- Combined Data ---
with combined_tab:
    st.header("🚴‍♂️🌤️ Explore Combined Dataset")
    st.markdown("Combining the cleaned bike-sharing and weather datasets to create a holistic view. "
                "This integration includes time-related and weather-based features, "
                "setting the stage for exploratory data analysis (EDA) and predictive modeling.")
    
    if step == "📋 Original Data":
        st.subheader("📋 Combined Dataset")

    elif step == "🧹 Data Cleaning":
        st.write("Let's clean the combined dataset.")

    else:
        st.write("The combined dataset is ready for analysis.")
