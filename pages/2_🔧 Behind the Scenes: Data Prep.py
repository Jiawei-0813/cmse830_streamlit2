import streamlit as st
import pandas as pd
import plotly.express as px

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

def display_features(features):
    """Display dataset feature descriptions."""
    for feature in features:
        st.markdown(f"- {feature}")

# --- Sidebar ---
step = st.sidebar.selectbox("Go to", ["🧹 Data Cleaning", "✅ Cleaned Data"])

# --- Main Content ---
bike_tab, weather_tab, combined_tab = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset", "Combined Dataset"])

# --- Bike Data ---
with bike_tab:
    st.header("🚴 Explore London Bike-Sharing Data")

    if step == "🧹 Data Cleaning":
        st.subheader("📋 Original Bike Data")
        st.dataframe(bike_0.head())
        st.markdown(f"""The original dataset contains **{bike_0.shape[0]:,}** rows and **{bike_0.shape[1]}** columns 
                    with both categorical and numerical features.""")

        st.markdown("### Data Quality Checks")
        cols1, cols2 = st.columns(2)
        with cols1:
            st.write(bike_0.dtypes.to_frame('Data Types'))

        with cols2:
            if st.checkbox("Check for Duplicates", key="bike_duplicates"):
                check_duplicates(bike_0)
            if st.checkbox("Check for Missing Data", key="bike_missing"):
                check_missing_data(bike_0)

        st.divider()
        
        st.subheader("🧹 Data Cleaning")
        st.markdown("Based on the data quality checks, we will now clean the dataset.")
        
        # Cleaning Process for Bike Data
        with st.expander("🔧 Data Adjustments and Cleaning Steps"):
            cleaning_bike = [
                "Converted `Start date` and `End date`: Changed to `datetime` format for consistency and analysis.",
                "Created `date` column: Extracted from `Start date` in `yyyy-mm-dd HH:MM` format for merging datasets.",
                "Converted `Total duration (ms)`: Transformed milliseconds to minutes for better interpretability.",
                "Dropped redundant columns: Removed `Total duration` as it duplicates information.",
                "Ensured consistency: Verified and aligned station names and station numbers.",
                "Optimized `Bike model`: Converted to categorical data type for better memory usage."
            ]
            display_features(cleaning_bike)

        # Copy of the original data for cleaning
        bike_1 = bike_0.copy()

        st.markdown("#### 1. Date and Time Adjustments")
        # Convert date columns to datetime
        bike_1["Start date"] = pd.to_datetime(bike_1["Start date"])
        bike_1["End date"] = pd.to_datetime(bike_1["End date"])

        # Extract date for merging
        bike_1['Date'] = bike_1['Start date'].dt.floor('T') # Round down to the nearest minute

        # Convert duration to minutes
        bike_1['Total duration (m)'] = round(bike_1['Total duration (ms)'] / 60000, 0)

        # Check the changes
        st.write(bike_1[['Start date', 'End date', 'Date', 'Total duration (ms)', 'Total duration (m)']].head())
        st.write(bike_1[['Start date', 'End date', 'Date', 'Total duration (ms)', 'Total duration (m)']].dtypes)

        # Drop redundant columns
        bike_1.drop(columns=['Start date', 'End date', 'Total duration (m)', 'Total duration (ms)'], inplace=True)

        st.markdown("#### 2. Station Consistency")
        # Consistency station names and numbers
        st.write("Unique values for Station Variables:")
        st.dataframe(bike_1[['Start station', 'Start station number', 'End station', 'End station number']].nunique().to_frame(name="Unique Values").transpose())

        # Group by name and number to identify mismatches
        station_mapping = bike_1.groupby(['Start station', 'Start station number']).size().reset_index(name='Count')

        # Highlight inconsistencies
        inconsistent_stations = station_mapping.groupby('Start station number').filter(lambda x: x['Start station'].nunique() > 1)
        if not inconsistent_stations.empty:
            st.warning(f"Inconsistent mappings found:\n{inconsistent_stations}")
        else:
            st.success("No inconsistencies found in station mappings.")
        
        st.subheader("🏙️ Station Popularity Analysis")

        # Station selection
        station_option = st.radio("Analyze station type:", ["Start station", "End station"])

        # Slider for selecting the number of top stations
        top_n = st.slider(f"Select top {station_option}s:", min_value=1, max_value=50, value=10)

        # Compute busiest stations
        top_stations = bike_0[station_option].value_counts().head(top_n).reset_index()
        top_stations.columns = [station_option, "Count"]

        # Bar plot for station popularity
        fig_station = px.bar(
            top_stations,
            x="Count",
            y=station_option,
            orientation="h",
            title=f"The Busiest {top_n} {station_option}s",
            labels={"Count": "Number of Bike-Sharing", station_option: "Station Name"},
            color=station_option,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_station)

        # Calculate percentage of trips covered
        total_trips = bike_0.shape[0]
        percentage = (top_stations["Count"].sum() / total_trips) * 100

        # Combine the explanation of the plot with the percentage calculation
        st.markdown(f"""
        This plot shows the **Number of Bike-Sharing Trips** (x-axis) for the **Busiest {top_n} {station_option.lower()}s** (y-axis).
        - Message: These busiest {station_option.lower()}s represent **{percentage:.2f}%** of the total bike-sharing trips.
        """)

        st.markdown("**🛠 Suggested Feature Engineering:**")
        st.markdown("""
        - **Clustering**: Use k-means or hierarchical clustering to group stations into clusters (e.g., high, medium, low usage).
        - **Encoding**: One-hot encode top 10 stations and group remaining stations into an "Other" category.
        """)

        # Convert bike model to categorical
        bike_1['Bike model'] = bike_1['Bike model'].astype('category')

    else:
        # Cleaned Bike Data
        st.subheader("✅ Cleaned Bike Data: Summary Statistics")
        numeric_vars, categorical_vars = categorize_var(bike_0)
        num_tab, cat_tab = st.tabs(["Numerical Variables", "Categorical Variables"])

        # Numerical Analysis
        with num_tab:
            st.subheader("Numerical Variables")
            if len(numeric_vars) > 0: 
                # Display descriptive statistics
                st.dataframe(bike_0[numeric_vars].describe().transpose().round(0))
                st.subheader("Outlier Detection")
                for num_var in numeric_vars:
                    outliers_count, lower, upper = check_outliers(bike_0, num_var)
            else:
                st.write("No numerical variables found in the dataset.")

        # Categorical Analysis
        with cat_tab:
            st.subheader("Categorical Variables")
            if len(categorical_vars) > 0:  
                unique_vals = bike_0[categorical_vars].nunique()
                st.dataframe(unique_vals.to_frame(name="Unique Values").transpose().style.format("{:,}").set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))

                selected_cat_var = st.selectbox(
                    "Select a categorical variable to analyze:", categorical_vars, key="bike_selected_cat"
                )
                if selected_cat_var:
                    value_counts = bike_0[selected_cat_var].value_counts()
                    st.markdown(f"### Value Counts for `{selected_cat_var}`")
                    st.bar_chart(value_counts)

# --- Weather Data ---
with weather_tab:
    st.header("🌤️ Explore London Weather Data")
  
    if step == "🧹 Data Cleaning":
        st.subheader("📋 Original Weather Data")
        st.dataframe(weather_0.head())
        st.markdown(f"""The original dataset contains **{weather_0.shape[0]:,}** rows and **{weather_0.shape[1]}** 
                    columns with all numerical features.""")

        st.markdown("### Data Quality Checks")
        cols1, cols2 = st.columns(2)
        with cols1:
            st.write(weather_0.dtypes.to_frame('Data Types'))

        with cols2:
            if st.checkbox("Check for Duplicates", key="weather_duplicates"):
                check_duplicates(weather_0)
            if st.checkbox("Check for Missing Data", key="weather_missing"):
                check_missing_data(weather_0)

        st.divider()
        
        st.subheader("🧹 Data Cleaning")
        st.markdown("Based on the data quality checks, we will now clean the dataset.")
        
        # Cleaning Process for Weather Data
        with st.expander("🔧 Data Adjustments and Cleaning Steps"):
            cleaning_weather = [
                "Converted `date`: Changed to `datetime` format for consistent time-based analysis.",
                "Removed timezone: Standardized time data by eliminating timezone differences.",
                "Extracted `Date`: Created a `yyyy-mm-dd HH:MM` format for merging with bike data.",
                "Mapped `weather_code`: Translated numeric codes into readable weather descriptions.",
                "Normalized numerical columns: Scaled values like temperature and wind speed for better comparisons."
            ]
            display_features(cleaning_weather)
        
        # Copy of the original data for cleaning
        weather_1 = weather_0.copy()

        # Convert date columns to datetime
        weather_1["date"] = pd.to_datetime(weather_1["date"])

        # Remove timezone
        weather_1["date"] = weather_1["date"].dt.tz_localize(None)

        # Extract date for merging
        weather_1['Date'] = weather_1['date'].dt.floor('T')

        # Map weather codes to descriptions



    else:
        # Cleaned Weather Data
        st.subheader("✅ Cleaned Weather Data: Summary Statistics ")
        numeric_vars, categorical_vars = categorize_var(weather_0)
        num_tab, cat_tab = st.tabs(["Numerical Variables", "Categorical Variables"])

        # Numerical Analysis
        with num_tab:
            st.subheader("Numerical Variables")
            if len(numeric_vars) > 0: 
                # Display descriptive statistics
                st.dataframe(weather_0[numeric_vars].describe().transpose().round(0))
                st.subheader("Outlier Detection")
                for num_var in numeric_vars:
                    outliers_count, lower, upper = check_outliers(weather_0, num_var)
                else:
                    st.write("No numerical variables found in the dataset.")


# --- Combined Data ---
with combined_tab:
    st.header("🚴‍♂️🌤️ Explore Combined Dataset")
    st.markdown("Combining the cleaned bike-sharing and weather datasets to create a holistic view. "
                "This integration includes time-related and weather-based features, "
                "setting the stage for exploratory data analysis (EDA) and predictive modeling.")
    
    st.markdown("### Steps in Merging:")
    with st.expander("🔗 Merge Workflow"):
        st.markdown("""
        1. Ensure consistent datetime formats in both datasets.
        2. Align granularity to hourly data (round to nearest hour).
        3. Perform an inner join on the `Date` column.
        4. Validate the merged dataset for missing or inconsistent rows.
        """)

    if step == "📋 Original Data":
        st.subheader("📋 Combined Dataset")

    elif step == "🧹 Data Cleaning":
        st.write("Let's clean the combined dataset.")

        st.markdown("### Features Added:")
        with st.expander("🎨 Feature Engineering Highlights"):
            st.markdown("""
            - **Time-Based Features**:
                - `Day of Week`: Weekday or weekend.
                - `Hour of Day`: Morning, afternoon, or evening.
            - **Weather-Based Features**:
                - `Temperature Category`: Cold, mild, or hot.
                - `Wind Speed Category`: Calm or windy.
            - **Combined Metrics**:
                - Bike counts categorized by weather conditions.
            """)
        # Combined Data
        with st.expander("✅ Combined Dataset"):
            st.dataframe(combined_data.head())  # Combined data
            st.markdown(f"Rows: **{combined_data.shape[0]:,}**, Columns: **{combined_data.shape[1]}**")
            st.markdown("""
            **New Features Added:**
            - Day of Week
            - Time of Day
            - Is Weekend
            """)

    else:        
        # Cleaned Bike Data
        st.subheader("✅ Ready Data: Summary Statistics")
        numeric_vars, categorical_vars = categorize_var(bike_0)
        num_tab, cat_tab = st.tabs(["Numerical Variables", "Categorical Variables"])

        # Numerical Analysis
        with num_tab:
            st.subheader("Numerical Variables")
            if len(numeric_vars) > 0: 
                # Display descriptive statistics
                st.dataframe(bike_0[numeric_vars].describe().transpose().round(0))
                st.subheader("Outlier Detection")
                for num_var in numeric_vars:
                    outliers_count, lower, upper = check_outliers(bike_0, num_var)
            else:
                st.write("No numerical variables found in the dataset.")

        # Categorical Analysis
        with cat_tab:
            st.subheader("Categorical Variables")
            if len(categorical_vars) > 0:  
                unique_vals = bike_0[categorical_vars].nunique()
                st.dataframe(unique_vals.to_frame(name="Unique Values").transpose().style.format("{:,}").set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))

                selected_cat_var = st.selectbox(
                    "Select a categorical variable to analyze:", categorical_vars, key="combined_selected_cat"
                )
                if selected_cat_var:
                    value_counts = bike_0[selected_cat_var].value_counts()
                    st.markdown(f"### Value Counts for `{selected_cat_var}`")
                    st.bar_chart(value_counts)

# Footer
st.markdown("### Ready to Explore")
st.write("With clean and merged data, we're set to uncover insights about London's bike-sharing trends!")

