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

# --- Main Content ---
bike_tab, weather_tab, combined_tab = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset", "Combined Dataset"])

# --- Bike Data ---
with bike_tab:
    st.header("🚴 Explore London Bike-Sharing Data")

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
    with st.expander("🔧 **Preprocessing Plan**", expanded = True):
        cleaning_bike = [
        "Convert `Start date` and `End date` to `datetime` for consistency.",
        "Extract `date` column from `Start date` in `yyyy-mm-dd HH:MM` format for merging.",
        "Convert `Total duration (ms)` to minutes for clarity.",
        "Drop redundant columns like `Total duration`.",
        "Verify and align station names and numbers for consistency.",
        "Convert Bike model from text (object) to numeric format using label encoding."
    ]
        display_features(cleaning_bike)

    # Copy of the original data for cleaning
    bike_1 = bike_0.copy()

    # Drop irrelevant columns
    bike_1.drop(columns=['Number', 'Bike number'], inplace=True)

    with st.expander("#### 1. Date Adjustments"):
        # Convert date columns to datetime
        bike_1["Start date"] = pd.to_datetime(bike_1["Start date"])
        bike_1["End date"] = pd.to_datetime(bike_1["End date"])

        # Extract date for merging
        bike_1['Date'] = bike_1['Start date'].dt.floor('T') # Round down to the nearest minute

        # Check the changes
        st.write("**Updated columns check**:")
        st.write(bike_1[['Start date', 'End date', 'Date']].head())

        st.write("**Data Types after Conversion**:")
        st.table(bike_1[['Start date', 'End date', 'Date']].dtypes.to_frame('Data Types').transpose())

        # Drop redundant columns
        bike_1.drop(columns=['Start date', 'End date'], inplace=True)
        st.success("Redundant variables has been dropped after checking.")
    
    with st.expander("#### 2. Total Duration and Outliers"):
        # Convert duration to minutes and remove zero values
        bike_1['Total duration (m)'] = round(bike_1['Total duration (ms)'] / 60000, 0)
        st.write(bike_1['Total duration (m)'].describe().round(0).to_frame().T)
        # Drop redundant columns
        bike_1.drop(columns=['Total duration (ms)', 'Total duration'], inplace=True)
        st.success("Converted total duration to minutes and dropped redundant columns.")

        # Number of trips with durations less than 1 minute
        less_than_1_min_count = (bike_1['Total duration (m)'] < 1).sum()
        less_than_1_min_percent = (less_than_1_min_count / bike_1.shape[0]) * 100

        st.write("**Initial Observations:**")
        initial_observations = [
            f"Max trip duration: **{bike_1['Total duration (m)'].max():,.0f} minutes** (potential outliers).",
            f"Trips < 1 minute: **{less_than_1_min_count:,} trips ({less_than_1_min_percent:.2f}%)**.",
            f"High variability: Std dev = **{bike_1['Total duration (m)'].std():,.0f} minutes**, Mean = **{bike_1['Total duration (m)'].mean():,.0f} minutes**."
        ]
        display_features(initial_observations)

        # Remove trips < 1 minute and outliers
        bike_2 = bike_1.copy()
        outliers_count, lower_bound, upper_bound = check_outliers(bike_1, 'Total duration (m)')
        bike_2 = bike_1[(bike_1['Total duration (m)'] > 1) & 
                        (bike_1['Total duration (m)'] >= lower_bound) & 
                        (bike_1['Total duration (m)'] <= upper_bound)]
        st.success(f"Dropped {less_than_1_min_count:,} trips < 1 minute and {outliers_count:,} outliers detected with the IQR approach.")

        # Min-Max Scaling
        scaler = MinMaxScaler()
        bike_2['Total duration (m)_scaled'] = scaler.fit_transform(bike_2[['Total duration (m)']])
        st.success("Apply Min-Max scaling to normalize durations to a [0, 1] range but retain relative differences.")
        
        # Create Subplots for Before and After Comparison
        fig = make_subplots(rows=1, cols=3, subplot_titles=(
            "Raw Data: Total Duration (Minutes)", 
            "After Outlier Removal", 
            "After Scaling (Min-Max)"
        ))

        # Boxplot for Raw Data
        fig.add_trace(
            go.Box(
                y=bike_1['Total duration (m)'],
                marker_color="skyblue",
                name="Raw Data"
            ),
            row=1, col=1
        )

        # Boxplot for Outlier Removed Data
        fig.add_trace(
            go.Box(
                y=bike_2['Total duration (m)'],
                marker_color="lightgreen",
                name="After Outlier Removal"
            ),
            row=1, col=2
        )

        # Boxplot for Scaled Data
        fig.add_trace(
            go.Box(
                y=bike_2['Total duration (m)_scaled'],
                marker_color="lightcoral",
                name="After Scaling"
            ),
            row=1, col=3
        )

        # Update layout to customize titles and appearance
        fig.update_layout(
            title="Comparison of Total Duration (m) Cleaning and Scaling",
            height=600,
            width=1000,
            template="simple_white",
            xaxis=dict(title=None, showgrid=False),
            yaxis=dict(title="Total Duration (Minutes)"),
            showlegend=False
        )

        # Display the plot
        st.plotly_chart(fig)

        # Insights 
        st.markdown("""
        ### Summary of Duration Cleaning and Scaling:
        1. **Filtered Short Trips (<1 Minute)**: Removed unrealistic durations likely caused by errors or anomalies.
        2. **Identified and Removed Outliers (IQR Method)**: Addressed extreme values to ensure a more reliable and interpretable dataset.
        3. **Applied Min-Max Scaling**: Normalized trip durations to a [0, 1] range, ensuring compatibility with machine learning models that require scaled inputs.
        4. **Prepared Cleaned Data**: Focused on realistic durations, creating a robust foundation for analysis and modeling.
        """)

    with st.expander("#### 3. Station Consistency & Popularity"):
        # Check consistency between station names and numbers
        st.subheader("Consistency Check for Station Names and Numbers")
        station_mapping = bike_2.groupby(['Start station number', 'Start station']).size().reset_index(name='Count')
        inconsistent_stations = station_mapping.groupby('Start station number').filter(lambda x: x['Start station'].nunique() > 1)

        if not inconsistent_stations.empty:
            st.warning("Inconsistencies found in station name and number mappings:")
            st.dataframe(inconsistent_stations)
        else:
            bike_2.drop(columns=['Start station number', 'End station number'], inplace=True)
            st.success("Station name and number mappings are consistent. Removing redundant station number columns.")
        
        # Unique values for station-related variables
        st.subheader("Unique Station Insights")
        cols1, cols2, cols3 = st.columns(3)
        with cols1:
            unique_start_stations = bike_2['Start station'].nunique()
            st.metric(label="Unique Start Stations", value=unique_start_stations)
        with cols2:
            unique_end_stations = bike_2['End station'].nunique()
            st.metric(label="Unique End Stations", value=unique_end_stations)
        with cols3:
            # Add a column for same start and end station
            bike_2['Same Start-End'] = (bike_2['Start station'] == bike_2['End station'])

            # Display metric for trips with same start and end station
            same_start_end_count = bike_2['Same Start-End'].sum()
            st.metric(label="Trips with Same Start and End Station", value=f"{same_start_end_count:,}")

        # Sample Data: Same Start-End Stations
        st.subheader("Sample of Start-End Station Pairs")
        st.dataframe(bike_2[['Start station', 'End station', 'Same Start-End']].sample(10))  # Shows random rows

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
            same_start_end_data = bike_2[bike_2["Same Start-End"]]
            top_stations = same_start_end_data["Start station"].value_counts().head(top_n).reset_index()
            top_stations.columns = ["Station Name", "Count"]
            title = f"Most Popular Same Start-End Stations"
        else:
            # Handle "Either Start or End" option
            if station_option == "Either Start or End":
                start_counts = bike_2["Start station"].value_counts()
                end_counts = bike_2["End station"].value_counts()
                combined_counts = start_counts.add(end_counts, fill_value=0).sort_values(ascending=False).head(top_n).reset_index()
                combined_counts.columns = ["Station Name", "Count"]
                top_stations = combined_counts
                title = f"Most popular {top_n} Stations by Either Start or End"
            else:
                # For Start or End Station
                top_stations = bike_2[station_option].value_counts().head(top_n).reset_index()
                top_stations.columns = [station_option, "Count"]
                title = f"Most popular {top_n} {station_option}s"

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
            total_trips = bike_2.shape[0]
        percentage = (top_stations["Count"].sum() / total_trips) * 100

        # Explanation of the plot
        st.markdown(f"""
        This plot shows the **Number of Bike-Sharing Trips** (x-axis) for the **most popular {top_n} {station_option.lower()}s** (y-axis).
        - It highlights the top stations where bike-sharing activity is the highest.
        - These {station_option.lower()}s represent **{percentage:.2f}%** of the total bike-sharing trips.
        """)

        st.divider()

        st.subheader(" 🚀 Suggested Feature Engineering:")
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

        st.markdown("**Key Observations**")
        bike_model_summary = [
            f"`CLASSIC` bikes account for **{round((716639 / (716639 + 59888)) * 100, 2)}%** of the dataset.",
            f"`PBSC_EBIKE` makes up the remaining **{round((59888 / (716639 + 59888)) * 100, 2)}%**.",
             "This highlights a significant imbalance, with far fewer e-bikes compared to classic bikes."
        ]
        display_features(bike_model_summary)

        # Label Encoding rationale
        st.subheader("Label Encoding Rationale")
 
        bike_model_reasons = [
            "There are only **two unique categories**, making label encoding straightforward and effective.",
            "It optimizes memory usage by replacing text labels with numeric values.",
            "Machine learning models often require numeric input for categorical data."
        ]
        display_features(bike_model_reasons)

        st.markdown("""
        The encoding will be as follows:
        - `CLASSIC` → 0
        - `PBSC_EBIKE` → 1
        """)

        # Apply Label Encoding
        try:
            le = LabelEncoder()
            bike_2['Bike Model Encoded'] = le.fit_transform(bike_2['Bike model'])
            st.success("Label encoding applied successfully!")
            st.dataframe(bike_2[['Bike model', 'Bike Model Encoded']].tail(6).T)

            # Drop original column
            bike_2.drop(columns=['Bike model'], inplace=True)
            st.success("Original `Bike model` column dropped after encoding.")
        except Exception as e:
            st.error(f"Label encoding failed. Error: {e}")

        st.divider()
        # Suggested next steps for feature engineering
        st.markdown("### 🚀 Suggested Feature Engineering")
        feature_engineering_steps = [
            "**Address Imbalance**: Consider SMOTE to balance the dataset if imbalance impacts model performance.",
            "**Explore Bike Types**: Investigate whether bike type affects trip duration, popular routes, or weather conditions.",
        ]
        display_features(feature_engineering_steps)

    st.divider()

    # Store cleaned data in session state
    st.subheader("✅ Cleaned Bike Data")
    if 'bike_2' not in st.session_state:
        st.session_state['bike_2'] = bike_2
    # st.write(st.session_state)
    st.success("Dataset cleaned and stored successfully.")
    st.dataframe(bike_2.sample(n=10))
    st.markdown(f"Rows: **{bike_2.shape[0]:,}**, Columns: **{bike_2.shape[1]}**")
    st.write(bike_2.dtypes.transpose())

# --- Weather Data ---
with weather_tab:
    st.header("🌤️ Explore London Weather Data")
  
    
    st.subheader("📋 Original Weather Data")
    st.dataframe(weather_0.head())
    st.markdown(f"""The original dataset contains **{weather_0.shape[0]:,}** rows and **{weather_0.shape[1]}** 
                columns with mostly numerical features.""")

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
    with st.expander("🔧 **Preprocessing Plan**", expanded=True):
        cleaning_weather = [
            "Convert `date` to `datetime` for consistency.",
            "Remove timezone to standardize time data.",
            "Extract `Date` in `yyyy-mm-dd HH:MM` format for merging.",
            "Map `weather_code` to readable weather descriptions.",
            "Normalize numerical columns like temperature and wind speed for better comparability."
        ]
        display_features(cleaning_weather)
    
    # Copy of the original data for cleaning
    weather_1 = weather_0.copy()

    with st.expander("#### 1. Date Adjustments"):
        # Convert date columns to datetime
        weather_1["date"] = pd.to_datetime(weather_1["date"])

        # Remove timezone
        weather_1["date"] = weather_1["date"].dt.tz_localize(None)

        # Extract date for merging
        weather_1['Date'] = weather_1['date'].dt.floor('T')

        # Provide a summary of changes
        st.success("Date adjustments completed:")
        adjustments = [
            "Converted `date` column to datetime format.",
            "Removed timezone information to standardize timestamps.",
            "Extracted precise timestamps (`Date`) rounded to the nearest minute for merging."
        ]
        display_features(adjustments)

        # Display data types of adjusted columns
        st.markdown("**Data Types After:**")
        st.dataframe(weather_1[['date', 'Date']].dtypes.to_frame(name="Data Type").transpose())

        # Drop redundant columns
        weather_1.drop(columns=['date'], inplace=True)      
        st.success("Redundant `date` column dropped after checking.")

    with st.expander("#### 2. Weather Code Preprocessing"):

        # convert weather code 
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

        # Copy for mapping
        weather_2 = weather_1.copy()

        weather_2['Weather Description'] = weather_2['weather_code'].map(weather_code_mapping)

        # Check for unmapped codes
        unmatched_codes = weather_2['weather_code'][~weather_1['weather_code'].isin(weather_code_mapping.keys())].unique()
        if len(unmatched_codes) > 0:
            st.warning(f"Unmatched weather codes found: {unmatched_codes}")

        # Count the occurrences of each weather description
        weather_counts = weather_2['Weather Description'].value_counts().reset_index()
        weather_counts.columns = ['Weather Description', 'Count']

        # Create an interactive bar plot using Plotly
        fig = px.bar(
            weather_counts,
            x='Weather Description',
            y='Count',
            title='Weather Description Distribution',
            labels={'Weather Description': 'Weather Condition', 'Count': 'Number of Occurrences'},
            color='Count',
            color_continuous_scale='Viridis'
        )

        # Display the plot
        st.plotly_chart(fig)

        # Drop redundant columns
        weather_2.drop(columns=['weather_code'], inplace=True)
    
    with st.expander("#### 3. Numerical Column Preprocessing"):
        # Describe initial statistics
        numeric_columns = ['temperature_2m', 'apparent_temperature', 'wind_speed_10m', 'relative_humidity_2m']
        st.subheader("Initial Summary Statistics")
        st.write(weather_2[numeric_columns].describe().T.round(2))

        # Outlier Detection and Removal
        st.subheader("Outlier Detection using the IQR Approach")
        for variable in numeric_columns:
            if variable == 'relative_humidity_2m':
                st.write(f"- `{variable}` is already in a reasonable range [0, 100]. No outliers removed.")
            else:
                original_size = weather_1.shape[0]
                _, lower_bound, upper_bound = check_outliers(weather_2, variable)
                weather_2 = weather_2[
                    (weather_2[variable] >= lower_bound) & (weather_2[variable] <= upper_bound)
                ]
                removed_count = original_size - weather_2.shape[0]
                st.write(f"- `{variable}`: {removed_count:,} records removed as outliers.")

        # Min-Max Scaling
        st.subheader("Min-Max Scaling")
        for variable in numeric_columns:
            scaler = MinMaxScaler()
            weather_2[f"{variable}_scaled"] = scaler.fit_transform(weather_2[[variable]])
        st.success("Min-Max scaling applied to all numerical columns.")

        # Visualization Before and After Scaling
        variable = st.selectbox("Select a variable to visualize", numeric_columns)
        if variable:
            fig_combined = make_subplots(
                rows=1, cols=2
            )

            # Add plots to the subplots
            fig_combined.add_trace(
                go.Box(y=weather_1[variable], name="Raw", marker_color="skyblue"), row=1, col=1
            )
            fig_combined.add_trace(
                go.Box(y=weather_2[f"{variable}_scaled"], name="After Scaling", marker_color="lightgreen"), row=1, col=2
            )

            # Update layout
            fig_combined.update_layout(
                title=f"Comparison for `{variable}`",
                height=600,
                width=1200,
                showlegend=False
            )

            # Display the combined plot
            st.plotly_chart(fig_combined)
        
        # Summary of changes
        st.subheader(f"Check Changes:")
        comparison_df = pd.concat(
            [
                weather_1[variable].describe().round(2),
                weather_2[f"{variable}_scaled"].describe().round(2)
            ],
            axis=1
        )

        st.write(comparison_df.transpose())

        # Drop original columns
        weather_2.drop(columns=numeric_columns, inplace=True)
        st.success("Original numerical columns dropped after scaling.")

        st.subheader("Process Overview")
        insights = [
            "Outliers have been identified and removed using the IQR method.",
            "Numerical columns have been normalized to [0, 1] for better comparability.",
            "Providing before-and-after comparisons through visualizations and statistics.",
            "Original columns were dropped to streamline the dataset."
        ]
        display_features(insights)
    
    # Store cleaned data in session state
    st.subheader("✅ Cleaned Weather Data")
    if 'weather_2' not in st.session_state:
        st.session_state['weather_2'] = weather_2
    # st.write(st.session_state)
    st.success("Dataset cleaned and stored successfully.")
    st.dataframe(weather_2.sample(n=10))
    st.markdown(f"Rows: **{weather_2.shape[0]:,}**, Columns: **{weather_2.shape[1]}**")
    st.table(weather_2.dtypes.to_frame('Data Types').transpose())

# --- Combined Data ---
with combined_tab:
    st.header("🚴‍♂️🌤️ Explore Combined Dataset")
    st.markdown("Combining the cleaned bike-sharing and weather datasets to create a holistic view. "
                "This integration includes time-related and weather-based features, "
                "setting the stage for exploratory data analysis (EDA) and predictive modeling.")
    
    st.markdown("### Steps in Merging:")
    if 'bike_2' in st.session_state:
        bike_2 = st.session_state['bike_2']
    if 'weather_2' in st.session_state:
        weather_2 = st.session_state['weather_2']
    
    # Merge Plan
    combined = pd.merge(bike_2, weather_2, on='Date', how='inner')

    with st.expander("🔗 Merge Workflow"):
        st.markdown("""
        1. Ensure consistent datetime formats in both datasets.
        2. Align granularity to hourly data (round to nearest hour).
        3. Perform an inner join on the `Date` column.
        4. Validate the merged dataset for missing or inconsistent rows.
        """)

    # Merge the cleaned datasets                                
    st.subheader("📋 Combined Dataset")

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
    combined_data = combined  # Define combined_data
    with st.expander("✅ Combined Dataset"):
        st.dataframe(combined_data.head())  # Combined data
        st.markdown(f"Rows: **{combined_data.shape[0]:,}**, Columns: **{combined_data.shape[1]}**")
        st.markdown("""
        **New Features Added:**
        - Day of Week
        - Time of Day
        - Is Weekend
        """)
    # Cleaned Combined Data
    st.subheader("✅ Ready Data: Summary Statistics")
    numeric_vars, categorical_vars = categorize_var(combined_data)
    num_tab, cat_tab = st.tabs(["Numerical Variables", "Categorical Variables"])

    # Numerical Analysis
    with num_tab:
        st.subheader("Numerical Variables")
        if len(numeric_vars) > 0: 
            # Display descriptive statistics
            st.dataframe(combined_data[numeric_vars].describe().transpose().round(0))
            st.subheader("Outlier Detection")
            for num_var in numeric_vars:
                outliers_count, lower, upper = check_outliers(combined_data, num_var)
                st.write(f"Outliers in `{num_var}`: {outliers_count}")
        else:
            st.write("No numerical variables found in the dataset.")

    # Categorical Analysis
    with cat_tab:
        st.subheader("Categorical Variables")
        if len(categorical_vars) > 0:  
            unique_vals = combined_data[categorical_vars].nunique()
            st.dataframe(unique_vals.to_frame(name="Unique Values").transpose().style.format("{:,}").set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))

            selected_cat_var = st.selectbox(
                "Select a categorical variable to analyze:", categorical_vars, key="combined_selected_cat"
            )
            if selected_cat_var:
                value_counts = combined_data[selected_cat_var].value_counts()
                st.markdown(f"### Value Counts for `{selected_cat_var}`")
                st.bar_chart(value_counts)
        else:
            st.write("No categorical variables found in the dataset.")

# Footer
st.markdown("### Ready to Explore")
st.write("With clean and merged data, we're set to uncover insights about London's bike-sharing trends!")

