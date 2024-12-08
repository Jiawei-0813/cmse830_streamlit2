import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler



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
        with st.expander("🔧 **Data Adjustments and Cleaning Steps**", expanded = True):
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

        with st.expander("#### 1. Date Adjustments"):
            # Convert date columns to datetime
            bike_1["Start date"] = pd.to_datetime(bike_1["Start date"])
            bike_1["End date"] = pd.to_datetime(bike_1["End date"])

            # Extract date for merging
            bike_1['Date'] = bike_1['Start date'].dt.floor('T') # Round down to the nearest minute

            # Check the changes
            st.write("**Updated columns check**:")
            st.write(bike_1[['Start date', 'End date', 'Date']].head())
            st.table(bike_1[['Start date', 'End date', 'Date']].dtypes.to_frame('Data Types').transpose())

            # Drop redundant columns
            bike_1.drop(columns=['Start date', 'End date'], inplace=True)
            st.success("Redundant variables has been dropped after checking.")
        
        with st.expander("#### 2. Total Duration Analysis"):
            # Step 1: Convert duration to minutes and remove zero values
            bike_1['Total duration (m)'] = round(bike_1['Total duration (ms)'] / 60000, 0)
            st.write(bike_1['Total duration (m)'].describe().round(2).to_frame().T)

            # Number of trips with durations less than 1 minute
            less_than_1_min_count = (bike_1['Total duration (m)'] < 1).sum()
            less_than_1_min_percent = (less_than_1_min_count / bike_1.shape[0]) * 100

            st.write("**Initial Observations:**")
            initial_observations = [
                f"Maximum trip duration is {bike_1['Total duration (m)'].max():.0f} minutes, indicating potential outliers.",
                f"{less_than_1_min_count:,} trips ({less_than_1_min_percent:.2f}%) have durations less than 1 minute."
                f"A standard deviation of {bike_1['Total duration (m)'].std():.0f} minutes compared to a mean of {bike_1['Total duration (m)'].mean():.0f} minutes indicates highly dispersed trip durations."
            ]
            display_features(initial_observations)

            # Remove trips with durations less than 1 minute
            bike_2 = bike_1[bike_1['Total duration (m)'] > 0]
            st.success("Trips with duration less than 1 minute have been removed.")

            # Step 2: Outlier detection and removal using IQR approach
            outliers_count, lower_bound, upper_bound = check_outliers(bike_1, 'Total duration (m)')
            bike_2 = bike_1[(bike_1['Total duration (m)'] >= lower_bound) &
                            (bike_1['Total duration (m)'] <= upper_bound)]
            st.success(f"{outliers_count} outliers removed using IQR.")

            # Step 3: Min-Max Scaling
            scaler = MinMaxScaler()
            bike_2['Total duration (m)_scaled'] = scaler.fit_transform(bike_2[['Total duration (m)']])
            st.success("Min-Max scaling applied successfully!")

            # Step 4: Create Subplots for Before and After Comparison
            fig = make_subplots(rows=1, cols=2, subplot_titles=(
                "Raw Data", 
                "After Cleaning"
            ))

            # Boxplot for Original Data
            fig.add_trace(
                go.Box(y=bike_1['Total duration (m)'], name="Before Cleaning", marker_color="skyblue"),
                row=1, col=1
            )

            # Boxplot for Cleaned Data
            fig.add_trace(
                go.Box(y=bike_2['Total duration (m)'], name="After Cleaning", marker_color="lightgreen"),
                row=1, col=2
            )

            fig.update_layout(
                title="Boxplot Comparison: Before and After Outlier Removal",
                height=600,
                width=1000,
                template="simple_white"
            )

            st.plotly_chart(fig)

            # Step 5: Insights from Visualizations
            st.markdown("""
            ### Insights from Visualizations
            2. **After Cleaning**: The cleaned dataset focuses on realistic trip durations, improving data reliability.
            3. **Min-Max Scaling**: Compresses the range to [0, 1], preparing the data for machine learning.
            """)

        with st.expander("#### 3. Station Consistency & Popularity"):
            # Check consistency between station names and numbers
            st.subheader("Consistency Check")
            station_mapping = bike_1.groupby(['Start station number', 'Start station']).size().reset_index(name='Count')
            inconsistent_stations = station_mapping.groupby('Start station number').filter(lambda x: x['Start station'].nunique() > 1)

            if not inconsistent_stations.empty:
                st.warning("Inconsistencies found in station name and number mappings:")
                st.dataframe(inconsistent_stations)
            else:
                bike_1.drop(columns=['Start station number', 'End station number'], inplace=True)
                st.success("Station name and number mappings are consistent.")
            
            # Unique values for station-related variables
            cols1, cols2, cols3 = st.columns(3)
            with cols1:
                unique_start_stations = bike_1['Start station'].nunique()
                st.metric(label="Unique Start Stations", value=unique_start_stations)
            with cols2:
                unique_end_stations = bike_1['End station'].nunique()
                st.metric(label="Unique End Stations", value=unique_end_stations)
            with cols3:
                # Add a column for same start and end station
                bike_1['Same Start-End'] = (bike_1['Start station'] == bike_1['End station']) & \
                                        (bike_1['Start station number'] == bike_1['End station number'])

                # Display metric for trips with same start and end station
                same_start_end_count = bike_1['Same Start-End'].sum()
                st.metric(label="Trips with Same Start and End Station", value=same_start_end_count)

            # Display same start-end station sample
            st.subheader("TOP: Same Start-End Trips")
            st.dataframe(bike_1[['Start station', 'End station']].sample(10))

            # Station Popularity Analysis
            st.subheader("🏙️ Station Popularity Analysis")
            st.markdown("""
            Explore the most popular start and end stations to identify key hubs in London's bike-sharing network.
            """)

            # Station selection
            station_option = st.radio("Analyze station type:", ["Start station", "End station", "Same Start-End", "Either Start or End"])

            # Slider for selecting the number of top stations
            top_n = st.slider(f"Select top {station_option}s:", min_value=1, max_value=50, value=10)

            if station_option == "Same Start-End":
                # Filter data for trips with the same start and end stations
                same_start_end_data = bike_1[bike_1["Same Start-End"]]
                top_stations = same_start_end_data["Start station"].value_counts().head(top_n).reset_index()
                top_stations.columns = ["Station Name", "Count"]
                title = f"Most Popular Same Start-End Stations"
            else:
                # Handle "Either Start or End" option
                if station_option == "Either Start or End":
                    start_counts = bike_1["Start station"].value_counts()
                    end_counts = bike_1["End station"].value_counts()
                    combined_counts = start_counts.add(end_counts, fill_value=0).sort_values(ascending=False).head(top_n).reset_index()
                    combined_counts.columns = ["Station Name", "Count"]
                    top_stations = combined_counts
                    title = f"Top {top_n} Stations by Either Start or End"
                else:
                    # For Start or End Station
                    top_stations = bike_1[station_option].value_counts().head(top_n).reset_index()
                    top_stations.columns = [station_option, "Count"]
                    title = f"The Busiest {top_n} {station_option}s"

            # Create a bar plot for station popularity
            fig_station = px.bar(
                top_stations,
                x="Count",
                y="Station Name" if station_option in ["Same Start-End", "Either Start or End"] else station_option,
                orientation="h",
                title=title,
                labels={"Count": "Number of Bike-Sharing", station_option: "Station Name"},
                color="Station Name" if station_option in ["Same Start-End", "Either Start or End"] else station_option,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            # Update layout to hide the legend
            fig_station.update_layout(showlegend=False)

            # Display the plot in Streamlit
            st.plotly_chart(fig_station)

            # Calculate percentage of trips covered
            if station_option == "Same Start-End":
                total_trips = same_start_end_data.shape[0]
            else:
                total_trips = bike_1.shape[0]
            percentage = (top_stations["Count"].sum() / total_trips) * 100

            # Explanation of the plot
            st.markdown(f"""
            This plot shows the **Number of Bike-Sharing Trips** (x-axis) for the **Busiest {top_n} {station_option.lower()}s** (y-axis).
            - It highlights the top stations where bike-sharing activity is the highest.
            - These busiest {station_option.lower()}s represent **{percentage:.2f}%** of the total bike-sharing trips.
            """)

            st.markdown("**🛠 Suggested Feature Engineering:**")
            st.markdown("""
            - **Clustering**: Use k-means or hierarchical clustering to group stations into clusters (e.g., high, medium, low usage).
            - **Encoding**: One-hot encode top 10 stations and group remaining stations into an "Other" category.
            """)

        with st.expander("#### 4. Bike Model Optimization"):
            # Bike model distribution counts
            bike_model_counts = pd.Series({'CLASSIC': 716639, 'PBSC_EBIKE': 59888})
            bike_model_df = bike_model_counts.reset_index()
            bike_model_df.columns = ['Bike Model', 'Count']

            # Display distribution
            fig_pie = px.pie(
                bike_model_df,
                names='Bike Model',
                values='Count',
                title='Bike Model Distribution',
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Set3,
                color_discrete_map={'CLASSIC': '#1f77b4', 'PBSC_EBIKE': '#ff7f0e'},
            )
            fig_pie.update_traces(
                textinfo='percent+label', 
                textfont_size=12,
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}",
            )
            fig_pie.update_layout(
                title={'text': 'Bike Model Distribution', 'x': 0.5, 'xanchor': 'center'},
                showlegend=False
            )
            st.plotly_chart(fig_pie)

            # Label Encoding rationale
            st.subheader("Label Encoding")
            st.markdown("""The `Bike model` column has two unique categories: `CLASSIC` and `PBSC_EBIKE`, 
            **label encoding** is the ideal for transforming categorical values into numerical representations:
            - `CLASSIC` → 0
            - `PBSC_EBIKE` → 1
            """)

            try:
                le = LabelEncoder()
                bike_1['Bike Model Encoded'] = le.fit_transform(bike_1['Bike model'])
                st.success("Label encoding applied successfully!")
                st.dataframe(bike_1[['Bike model', 'Bike Model Encoded']].tail(6).T)

                # Drop original column
                bike_1.drop(columns=['Bike model', 'Bike number'], inplace=True)
                st.success("`Bike model` was successfully encoded and the original column has been dropped.")
            except Exception as e:
                st.error(f"Label encoding failed. Error: {e}")

            # Explanation of benefits
            st.markdown("""
            **Why Optimize?**
            - Reduces memory usage
            - Transforms categorical data into numeric format
            - Ensures compatibility with machine learning algorithms
            """)

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

        with st.expander("#### 1. Date and Time Adjustments"):
            # Convert date columns to datetime
            weather_1["date"] = pd.to_datetime(weather_1["date"])

            # Remove timezone
            weather_1["date"] = weather_1["date"].dt.tz_localize(None)

            # Extract date for merging
            weather_1['Date'] = weather_1['date'].dt.floor('T')

        with st.expander("#### 2. Map Weather Codes"):
            weather_code_mapping = {
                0: "Clear Sky",
                1: "Partly Cloudy",
                2: "Cloudy",
                3: "Overcast",
                45: "Fog",
                48: "Rime Fog",
                51: "Drizzle",
                53: "Moderate Drizzle",
                55: "Heavy Drizzle",
                61: "Slight Rain",
                63: "Moderate Rain",
                65: "Heavy Rain",
                71: "Light Snow",
                73: "Moderate Snow",
                75: "Heavy Snow",
                80: "Rain Showers",
                95: "Thunderstorm",
                96: "Thunderstorm with Hail"
            }

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

