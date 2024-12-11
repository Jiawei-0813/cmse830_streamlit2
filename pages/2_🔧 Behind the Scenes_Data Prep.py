import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide", 
               page_title="🔧 Behind the Scenes_Data Prep"
               )

st.title("🔧 Behind the Scenes: Data Prep")
st.markdown("""
This page is dedicated to the data preparation process.
- We’ll examine raw datasets, clean and merge them, and create new features for deeper insights.
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
bike_tab, weather_tab, combined_tab = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset", "🚴‍♂️🌤️Combined Dataset"])

# --- Bike Data ---
with bike_tab:
    st.header("🚴 Explore London Bike-Sharing Data")

    # Original Bike Data Overview
    st.subheader("📋 Original Bike Data")
    
    st.markdown(f"""The original dataset contains **{bike_0.shape[0]:,}** rows and **{bike_0.shape[1]}** columns 
                with both categorical and numerical features.""")
    st.dataframe(bike_0.head())
    
    # Data Quality Checks
    st.markdown("### 🔍 Data Quality Checks")
    st.markdown("""
    Checking for missing values and duplicates ensures the dataset is complete and reliable for analysis.
    """)

    cols1, cols2 = st.columns(2)
    with cols1:
        st.write(bike_0.dtypes.to_frame('Data Types'))

    with cols2:
        st.write("**Data Issues**")
        if st.checkbox("Check for Duplicates", key="bike_duplicates"):
            check_duplicates(bike_0)
        if st.checkbox("Check for Missing Data", key="bike_missing"):
            check_missing_data(bike_0)

    st.divider()
    
    # Data Cleaning Overview
    st.subheader("🧹 Preprocessing Workflow")
    with st.container():
        st.markdown("""
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0;">
            <h4 style="color: #4a7cfe; margin-top: 0;">Key Cleaning Steps</h4>
            <ul style="font-size: 14px; line-height: 1.6;">
                <li><strong>Standardize Dates:</strong> Convert <code>Start date</code> and <code>End date</code> to datetime for consistency.</li>
                <li><strong>Create Date Feature:</strong> Extract <code>Date</code> from <code>Start date</code> in <code>yyyy-mm-dd HH:MM</code> format for merging.</li>
                <li><strong>Convert Durations:</strong> Transform <code>Total duration (ms)</code> to minutes for readability.</li>
                <li><strong>Drop Redundancies:</strong> Remove unused columns like <code>Total duration</code>.</li>
                <li><strong>Verify Station Consistency:</strong> Align station names and numbers to maintain data integrity.</li>
                <li><strong>Encode Bike Models:</strong> Convert <code>Bike model</code> into numeric format using label encoding.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Copy of the original data for cleaning
    bike_1 = bike_0.copy()

    # Drop irrelevant columns
    bike_1.drop(columns=['Number', 'Bike number'], inplace=True)

    with st.expander("### 1️⃣ Adjusting Date Columns"):
        # Convert date columns to datetime
        bike_1["Start date"] = pd.to_datetime(bike_1["Start date"])
        bike_1["End date"] = pd.to_datetime(bike_1["End date"])
    
        # Extract date for merging
        bike_1['Date'] = bike_1['Start date'].dt.floor('T') # Round down to the nearest minute

        # Check the changes
        cols1, cols2 = st.columns([2,1])
        with cols1:
            st.write("**Updated Columns**:")
            st.dataframe(bike_1[['Start date', 'End date', 'Date']].head())
        with cols2:
            st.write("**Data Types after Conversion**:")
            st.dataframe(bike_1[['Start date', 'End date', 'Date']].dtypes.to_frame('Data Types'))
        st.success("Converted `Start date` and `End date` to datetime format and extracted `Date` for merging.")

        # Drop redundant columns
        bike_1.drop(columns=['Start date', 'End date'], inplace=True)
        st.success("Dropped redundant columns after verifying conversions.")
    
    with st.expander("### 2️⃣ Cleaning Trip Durations"):
        # Convert duration to minutes and remove zero values
        bike_1['Total duration (m)'] = round(bike_1['Total duration (ms)'] / 60000, 0)

        # Drop redundant columns
        bike_1.drop(columns=['Total duration (ms)', 'Total duration'], inplace=True)
        st.success("1. Converted `Total duration (ms)` to minutes and dropped redundant columns.")

        # Display summary statistics
        st.write("**Summary Statistics**")
        st.dataframe(bike_1["Total duration (m)"].describe().round(2).to_frame().T)

        # Identify short trips and calculate max duration in days
        less_than_1_min_count = (bike_1['Total duration (m)'] < 1).sum()
        less_than_1_min_percent = (less_than_1_min_count / bike_1.shape[0]) * 100
        max_duration_days = bike_1['Total duration (m)'].max() / (24 * 60)

        # Observations
        display_features([
            f"**Maximum Duration:** {bike_1['Total duration (m)'].max():,.0f} minutes (~{max_duration_days:.2f} days), indicating potential outliers.",
            f"**Short Trips (<1 Minute):** {less_than_1_min_count:,} trips ({less_than_1_min_percent:.2f}%) may indicate data errors or anomalies.",
            f"**High Variability:** Standard deviation of {bike_1['Total duration (m)'].std():,.0f} minutes reflects a wide range of trip durations."
        ])

        # Remove trips < 1 minute and outliers
        bike_2 = bike_1.copy()
        outliers_count, lower_bound, upper_bound = check_outliers(bike_1, 'Total duration (m)')
        bike_2 = bike_1[(bike_1['Total duration (m)'] > 1) & 
                        (bike_1['Total duration (m)'] >= lower_bound) & 
                        (bike_1['Total duration (m)'] <= upper_bound)]
        st.success(f"2. Removed **{less_than_1_min_count:,}** short trips (<1 min) and **{outliers_count:,}** outliers using the IQR approach.")

        # Apply Min-Max Scaling
        scaler = MinMaxScaler()
        bike_2['Total duration (m)_scaled'] = scaler.fit_transform(bike_2[['Total duration (m)']])
        st.success("3. Applied Min-Max scaling to normalize durations while retaining relative differences.")

        # Visualize cleaning and scaling process
        fig = make_subplots(rows=1, cols=3, subplot_titles=[
            " ", " ", " "
        ])
        fig.add_trace(go.Box(y=bike_1['Total duration (m)'], name="Raw Data", marker_color="skyblue"), row=1, col=1)
        fig.add_trace(go.Box(y=bike_2['Total duration (m)'], name="Outliers Removed", marker_color="lightgreen"), row=1, col=2)
        fig.add_trace(go.Box(y=bike_2['Total duration (m)_scaled'], name="Scaled Data", marker_color="lightcoral"), row=1, col=3)
        fig.update_layout(
            title="Duration Cleaning and Scaling Comparison",
            height=600, width=1000, template="simple_white",
            yaxis=dict(title="Total Duration (Minutes)"), showlegend=False
        )
        
        # Display the plot
        st.plotly_chart(fig)

        # Add explanation
        display_features([
            "The first plot shows raw trip durations, including outliers.",
            "The second plot displays the cleaned data after removing short trips and outliers.",
            "The final plot shows normalized trip durations using Min-Max scaling."
        ])

        st.divider()

        # Summary insights
        st.write("**Summary of Duration Cleaning and Scaling:**")
        key_takeaways = [
            "**Filtered Short Trips**: Removed trips less than 1 minute (likely anomalies).",
            "**Removed Outliers**: Addressed extreme values using the IQR method.",
            "**Min-Max Scaling**: Normalized data for better compatibility with machine learning models."
        ]
        display_features(key_takeaways)

    with st.expander("### 3️⃣ Validating Station Consistency"):
        # Station Names and Numbers Consistency Check
        st.markdown("**1. Verifying Station Name and Number Consistency**")
        station_mapping = bike_2.groupby(['Start station number', 'Start station']).size().reset_index(name='Count')
        inconsistent_stations = station_mapping.groupby('Start station number').filter(lambda x: x['Start station'].nunique() > 1)

        if not inconsistent_stations.empty:
            st.warning("Inconsistencies found in station name and number mappings:")
            st.dataframe(inconsistent_stations)
        else:
            bike_2.drop(columns=['Start station number', 'End station number'], inplace=True)
            st.success("Station names and numbers are consistent. Redundant columns removed.")

        st.divider()
        
        # Unique Station Insights
        st.markdown("**2. Unique Station Insights**")
        cols1, cols2, cols3 = st.columns(3)
        with cols1:
            unique_start_stations = bike_2['Start station'].nunique()
            st.metric(label="Start Stations (Unique)", value=unique_start_stations)
        with cols2:
            unique_end_stations = bike_2['End station'].nunique()
            st.metric(label="End Stations (Unique)", value=unique_end_stations)
        with cols3:
            # Add a column for same start and end station
            bike_2['Same Start-End'] = (bike_2['Start station'] == bike_2['End station'])

            # Display metric for trips with same start and end station
            same_start_end_count = bike_2['Same Start-End'].sum()
            st.metric(label="Same Start-End Trips", value=f"{same_start_end_count:,}")

        # Sample Data: Same Start-End Stations
        st.markdown("**Sample of Trips with Same Start-End Stations**")
        st.dataframe(bike_2[['Start station', 'End station', 'Same Start-End']].sample(10))  # Shows random rows

        # Suggested Feature Engineering
        st.markdown("### 🚀 Suggested Feature Engineering")
        st.markdown("""
        - **Clustering:** Use k-means or hierarchical clustering to group stations by usage levels (e.g., high, medium, low).  
        - **Encoding:** One-hot encode the top 10 stations and group the rest into an "Other" category.  
        """)

    with st.expander("### 4️⃣ Encoding Bike Models"):
        # Bike model distribution counts
        st.markdown("**1. Bike Model Distribution**")
        bike_model_counts = pd.Series({'CLASSIC': 716639, 'PBSC_EBIKE': 59888})
        bike_model_df = bike_model_counts.reset_index()
        bike_model_df.columns = ['Bike Model', 'Count']

        cols1, cols2 = st.columns(2)
        with cols1:
            # Display distribution
            fig_pie = px.pie(
            bike_model_df,
            names='Bike Model',
            values='Count',
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
            showlegend=False
            )
            st.plotly_chart(fig_pie)

        with cols2:
            st.markdown("**Observations**")
            bike_model_summary = [
            f"`CLASSIC` bikes account for **{round((716639 / (716639 + 59888)) * 100, 2)}%** of the dataset.",
            f"`PBSC_EBIKE` makes up the remaining **{round((59888 / (716639 + 59888)) * 100, 2)}%**.",
            "This highlights a significant imbalance, with far fewer e-bikes compared to classic bikes."
            ]
            display_features(bike_model_summary)

        st.divider()

        # Rationale for Label Encoding
        st.markdown("**2. Rationale for Label Encoding**")
 
        bike_model_reasons = [
            "The `Bike model` column contains only **two unique categories**, making label encoding straightforward and effective.",
            "Converting to numeric values optimizes memory usage and is necessary for machine learning models."
        ]
        display_features(bike_model_reasons)

        st.markdown("""
        **Encoding Mapping:**
        - `CLASSIC` → 0
        - `PBSC_EBIKE` → 1
        """)

        # Apply Label Encoding
        le = LabelEncoder()
        bike_2["Bike Model Encoded"] = le.fit_transform(bike_2["Bike model"])
        st.success("Successfully encoded `Bike model` as numeric.")

        # Display Encoded Sample
        st.markdown("**Sample of Encoded Bike Models:")
        st.dataframe(bike_2[["Bike model", "Bike Model Encoded"]].sample(5))
        bike_2.drop(columns=["Bike model"], inplace=True)
        st.success("Dropped the original `Bike model` column to maintain dataset cleanliness.")

        st.divider()

        # Suggested next steps for feature engineering
        st.markdown("**🚀 Suggested Next Steps**")
        feature_engineering_steps = [
            "**Addressing Imbalance:** Techniques like SMOTE can balance the dataset if needed for specific models.",
            "**Exploring Bike Models**: Investigate relationships between bike type and factors like trip duration, popular routes, or weather conditions.",
        ]
        display_features(feature_engineering_steps)

    st.divider()

    # Final Cleaned Data
    st.subheader("✅ Bike Cleaned Dataset")
    if 'bike_2' not in st.session_state:
        st.session_state['bike_2'] = bike_2
    # st.write(st.session_state)

    st.success("Bike dataset cleaned and stored successfully.")
    st.markdown("**Sample of Cleaned Data:**")
    st.dataframe(bike_2.sample(10))
    st.markdown(f"Rows: **{bike_2.shape[0]:,}**, Columns: **{bike_2.shape[1]}**")
    st.dataframe(bike_2.dtypes.to_frame('Data Types').transpose())

# --- Weather Data ---
with weather_tab:
    st.header("🌤️ Explore London Weather Data")
  
    st.subheader("📋 Original Weather Data")
    st.markdown(f"""The original dataset contains **{weather_0.shape[0]:,}** rows and **{weather_0.shape[1]}** 
            columns with mostly numerical features.""")
    st.dataframe(weather_0.head())

    st.markdown("### 🔍 Data Quality Checks")
    st.markdown("""
    Checking for missing values and duplicates ensures the dataset is complete and reliable for analysis.
    """)
    cols1, cols2 = st.columns(2)
    with cols1:
        st.write(weather_0.dtypes.to_frame('Data Types'))

    with cols2:
        st.markdown("**Data Issues**")
        if st.checkbox("Check for Duplicates", key="weather_duplicates"):
            check_duplicates(weather_0)
        if st.checkbox("Check for Missing Data", key="weather_missing"):
            check_missing_data(weather_0)

    st.divider()

    # Data Cleaning Overview
    st.subheader("🧹 Preprocessing Workflow")
    with st.container():
        st.markdown("""
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0;">
            <h4 style="color: #4a7cfe; margin-top: 0;">Key Cleaning Steps</h4>
            <ul style="font-size: 14px; line-height: 1.6;">
                <li><strong>Standardize Dates:</strong> Convert <code>date</code> column to datetime for consistency.</li>
                <li><strong>Create Date Feature:</strong> Extract <code>Date</code> in <code>yyyy-mm-dd HH:MM</code> format for merging.</li>
                <li><strong>Remove Timezone:</strong> Standardize timestamps by removing timezone information.</li>
                <li><strong>Map Weather Descriptions:</strong> Decode <code>weather_code</code> into human-readable weather conditions.</li>
                <li><strong>Normalize Numerical Columns:</strong> Scale variables like temperature and wind speed for better comparability.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    
    # Copy of the original data for cleaning
    weather_1 = weather_0.copy()

    with st.expander("### 1️⃣ Adjusting Date Columns"):
        # Convert date columns to datetime
        weather_1["date"] = pd.to_datetime(weather_1["date"])

        # Remove timezone
        weather_1["date"] = weather_1["date"].dt.tz_localize(None)

        # Extract date for merging
        weather_1['Date'] = weather_1['date'].dt.floor('T')

        cols1, cols2 = st.columns([1, 1.5])
        with cols1:
            # Display data types of adjusted columns
            st.markdown("**Data Types After Conversion:**")
            st.dataframe(weather_1[['date', 'Date']].dtypes.to_frame(name="Data Type"))

        with cols2:
            # Drop redundant columns
            weather_1.drop(columns=['date'], inplace=True)      

            st.success("Date adjustments completed:")
            adjustments = [
            "Converted `date` column to datetime format.",
            "Removed timezone information to standardize timestamps.",
            "Extracted `Date` rounded to the nearest minute for merging.",
            "Dropped redundant `date` column after adjustments."
            ]
            display_features(adjustments)

    with st.expander("### 2️⃣ Mapping Weather Codes"):
        # Convert weather_code 
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
        unmatched_codes = weather_2['weather_code'][~weather_2['weather_code'].isin(weather_code_mapping.keys())].unique()
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
            labels={'Weather Description': '', 'Count': 'Number of Occurrences'},
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        
        # Display the plot
        st.plotly_chart(fig)

        # Observations
        observations = [
            f"**Overcast** is the most frequent weather condition with **{weather_counts.iloc[0]['Count']} occurrences**.",
            f"The least frequent conditions are: **{', '.join(weather_counts['Weather Description'].iloc[-3:].tolist())}**.",
            "The data is heavily skewed toward a few weather conditions, like Overcast and Clear Sky."
        ]
        display_features(observations)

        # Drop redundant column
        weather_2.drop(columns=['weather_code'], inplace=True)
        st.success("Mapped `weather_code` to descriptive labels and dropped the original column.")

        st.divider()

        # Suggested Next Steps
        st.markdown("**🚀 Suggested Next Steps**")
        next_steps = [
            "Group similar weather types (e.g., Drizzle, Rain) for simpler analysis.",
            "Apply one-hot encoding for predictive modeling or label encoding if needed for numerical compatibility."
        ]
        display_features(next_steps)
    
    with st.expander("### 3️⃣ Normalizing Numerical Columns"):
        # Describe initial statistics
        numeric_columns = ['temperature_2m', 'apparent_temperature', 'wind_speed_10m', 'relative_humidity_2m']
        
        # Display initial summary statistics
        st.markdown("**1. Initial Summary Statistics**")
        st.write(weather_2[numeric_columns].describe().T.round(2))

        # Outlier Detection and Removal using IQR
        st.markdown("**2. Outlier Detection and Removal**")
        for variable in numeric_columns:
            if variable == 'relative_humidity_2m':
                st.write(f"- `{variable}`: No outliers removed as values are within the reasonable range [0, 100].")
            else:
                original_size = weather_1.shape[0]
                _, lower_bound, upper_bound = check_outliers(weather_2, variable)
                weather_2 = weather_2[
                    (weather_2[variable] >= lower_bound) & (weather_2[variable] <= upper_bound)
                ]
                removed_count = original_size - weather_2.shape[0]
                st.write(f"- `{variable}`: Remove {removed_count:,} outliers using the IQR approach.")


        # Min-Max Scaling
        st.markdown("**3. Min-Max Scaling**")
        for variable in numeric_columns:
            scaler = MinMaxScaler()
            weather_2[f"{variable}_scaled"] = scaler.fit_transform(weather_2[[variable]])
        st.success("Min-Max scaling applied to all numerical columns.")

        # Visualization: Raw Data vs. After Scaling
        st.markdown("**4. Visualization**")
        
        variable = st.selectbox("Select a variable to visualize:", numeric_columns)

        if variable:
            # Visualize cleaning and scaling process
            fig = make_subplots(rows=1, cols=3, subplot_titles=[
            " ", " ", " "
            ])
            fig.add_trace(go.Box(y=weather_1[variable], name="Raw Data", marker_color="skyblue"), row=1, col=1)
            fig.add_trace(go.Box(y=weather_2[variable], name="Outliers Removed", marker_color="lightgreen"), row=1, col=2)
            fig.add_trace(go.Box(y=weather_2[f"{variable}_scaled"], name="Scaled Data", marker_color="lightcoral"), row=1, col=3)
            fig.update_layout(
            height=600, width=1000, template="simple_white",
            showlegend=False
            )
            
            # Display the plot
            st.plotly_chart(fig)
        
        # Check Changes
        st.markdown("**Check Changes**")
        comparison_df = pd.concat(
            [
                weather_1[variable].describe().round(2),
                weather_2[variable].describe().round(2),
                weather_2[f"{variable}_scaled"].describe().round(2)
            ],
            axis=1
        )

        st.write(comparison_df.transpose())

        # Drop original columns
        weather_2.drop(columns=numeric_columns, inplace=True)
        st.success("Original numerical columns dropped after scaling.")

        st.divider()

        # Process Overview
        st.write("**Summary of Cleaning and Scaling:**")
        insights = [
            "Identified and removed outliers using the IQR method.",
            "Applied Min-Max scaling to normalize numerical columns.",
            "Visualized the impact of scaling using side-by-side plots.",
            "Dropped raw numerical columns after scaling for streamlined data."
        ]
        display_features(insights)
    
    st.divider()

    # Final Cleaned Data
    st.subheader("✅ Weather Cleaned Dataset")
    if 'weather_2' not in st.session_state:
        st.session_state['weather_2'] = weather_2
    # st.write(st.session_state)

    st.success("Weather dataset cleaned and stored successfully.")
    st.markdown("**Sample of Cleaned Data:**")
    st.dataframe(weather_2.sample(n=10))
    st.markdown(f"Rows: **{weather_2.shape[0]:,}**, Columns: **{weather_2.shape[1]}**")
    st.dataframe(weather_2.dtypes.to_frame('Data Types').transpose())

# --- Combined Data ---
with combined_tab:
    st.header("🚴🌤️ Combined Bike and Weather Data")
    st.markdown(
        "This step combines the cleaned bike-sharing and weather datasets to create a holistic view. "
        "This integration includes time-related and weather-based features, "
        "setting the stage for exploratory data analysis (EDA) and predictive modeling."
    )
    
    # Load cleaned datasets from session state
    if 'bike_2' in st.session_state:
        bike_2 = st.session_state['bike_2']
    else:
        st.error("Cleaned bike dataset not found in session state.")
        st.stop()

    if 'weather_2' in st.session_state:
        weather_2 = st.session_state['weather_2']
    else:
        st.error("Cleaned weather dataset not found in session state.")
        st.stop()
    
        # Merge Workflow
    with st.container():
        with st.container():
            st.markdown("""
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0;">
                <h4 style="color: #4a7cfe; margin-top: 0;">Merge Workflow</h4>
                <ul style="font-size: 14px; line-height: 1.6;">
                    <li><strong>Ensure Consistent Datetime Format:</strong> Both datasets should have a standardized datetime format.</li>
                    <li><strong>Perform Inner Join:</strong> Merge datasets on the <code>Date</code> column to align time periods.</li>
                    <li><strong>Validate Merged Dataset:</strong> Check for missing or inconsistent rows after merging.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        combined = pd.merge(bike_2, weather_2, on='Date', how='inner')
        st.success(f"Datasets successfully merged. Combined dataset contains **{combined.shape[0]:,} rows** and **{combined.shape[1]} columns**.")
    
    st.divider()

    # Feature Engineering
    st.subheader("🚦Feature Engineering")
   
    # Time-Based Features
    with st.expander("1️⃣ Time-Based Features", expanded=True):
        display_features([
            "Extracted time-based features from the `Date` column enables the analysis of temporal patterns:"
        ])

        st.success("Time-based features created successfully!")
        
        combined['Day_of_Month'] = combined['Date'].dt.day  # 1-31
        combined['Day_of_Week'] = combined['Date'].dt.dayofweek + 1  # Monday=1, Sunday=7
        combined['Hour_of_Day'] = combined['Date'].dt.hour  # 0-23
        combined['is_Weekend'] = combined['Day_of_Week'].apply(lambda x: 1 if x > 5 else 0)  # 1=Weekend, 0=Weekday
        combined['Time_of_Day'] = combined['Hour_of_Day'].apply(
            lambda hour: 'Morning' if 5 <= hour < 12 else
                         'Afternoon' if 12 <= hour < 17 else
                         'Evening' if 17 <= hour < 21 else 'Night'
        )
        st.dataframe(combined[['Date', 'Day_of_Week', 'Hour_of_Day', 'is_Weekend', 'Time_of_Day']].sample(5))

    # Weather-Based Features
    with st.expander("2️⃣ Weather-Based Features", expanded=True):
        display_features([
            "Some weather conditions account for only a small portion of the dataset.",
            "Consolidating them into a binary `is_Rainy` feature simplifies analysis and enhances interpretability."
        ])

        # Identify rainy conditions
        rain_conditions = ["Drizzle", "Moderate Drizzle", "Heavy Drizzle", 
                            "Slight Rain", "Moderate Rain", "Heavy Rain", 
                            "Rain Showers"]
        
        # Verify mapping consistency
        unmatched_weather = combined.loc[~combined['Weather Description'].isin(rain_conditions + ['Overcast', 'Partly Cloudy', 'Clear Sky', 'Cloudy']), 'Weather Description'].unique()
        if len(unmatched_weather) > 0:
            st.warning(f"Unmatched weather descriptions found: {unmatched_weather}")
        
        # Create `is_Rainy` binary feature
        combined['is_Rainy'] = combined['Weather Description'].isin(rain_conditions).astype(int)

        # Count of each Weather Description
        weather_counts = combined['Weather Description'].value_counts().reset_index()
        weather_counts.columns = ['Weather Description', 'Count']
        weather_counts['Category'] = weather_counts['Weather Description'].apply(
            lambda x: 'Rainy' if x in rain_conditions else 'Not Rainy'
        )

        # Cross-check Rainy/Not Rainy counts
        rainy_total = weather_counts.loc[weather_counts['Category'] == 'Rainy', 'Count'].sum()
        not_rainy_total = weather_counts.loc[weather_counts['Category'] == 'Not Rainy', 'Count'].sum()
        total_combined = combined.shape[0]

        if rainy_total + not_rainy_total != total_combined:
            st.warning(f"Mismatch in counts: Total={total_combined}, Rainy={rainy_total}, Not Rainy={not_rainy_total}")

        st.divider()

        # Display Weather Description Counts
        cols1, cols2 = st.columns(2)

        with cols1:
            st.markdown("**Count of Each Weather Description**")
            st.dataframe(weather_counts.style.format({'Count': '{:,}'}).set_properties(**{'text-align': 'center'}))

        with cols2:
            # Create a pie chart for `is_Rainy` feature
            is_rainy_dist = combined['is_Rainy'].value_counts().reset_index()
            is_rainy_dist.columns = ['Rainy Condition', 'Count']
            is_rainy_dist['Rainy Condition'] = is_rainy_dist['Rainy Condition'].replace({0: 'Not Rainy', 1: 'Rainy'})

            fig_pie = px.pie(
            is_rainy_dist,
            names='Rainy Condition',
            values='Count',
            hole=0.5,
            color_discrete_sequence=['#4CAF50', '#2196F3'],
            color_discrete_map={'Rainy': '#4CAF50', 'Not Rainy': '#2196F3'},
            )
            fig_pie.update_traces(
            textinfo='percent+label',
            textfont_size=12,
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}",
            )
            fig_pie.update_layout(
                title={
                    'text': "Distribution of Rainy vs Not Rainy Conditions",
                    'y': 0.05,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'bottom'
                },
                showlegend=False
            )
            st.plotly_chart(fig_pie)

        st.success("Created and verified `is_Rainy` binary feature successfully.")

    st.divider()

    # **Final Combined Dataset**
    st.subheader("✅ Final Combined Dataset")
    st.markdown(
        "The combined dataset is ready for EDA and modeling. Below is a summary of the dataset."
    )
    
    # Display sample rows
    st.dataframe(combined.sample(5))
    
    # Dataset summary
    cols1, cols2 = st.columns(2)
    with cols1:
        st.markdown(f"Rows: **{combined.shape[0]:,}**, Columns: **{combined.shape[1]}**")
    with cols2:
        if st.checkbox("Show Data Types"):
            st.write(combined.dtypes.to_frame(name='Data Type'))
    
    # Numerical Features Overview
    tab1, tab2 = st.tabs(["🔢 Numerical Features", "🅰️ Categorical Features"])

    with tab1:
        st.markdown("### Summary Statistics for Numerical Features")
        numeric_vars = [
            "Total duration (m)", "Total duration (m)_scaled", "Bike Model Encoded",
            "temperature_2m_scaled", "apparent_temperature_scaled",
            "wind_speed_10m_scaled", "relative_humidity_2m_scaled",
            "Day_of_Month", "Day_of_Week", "Hour_of_Day"
        ]
        st.dataframe(combined[numeric_vars].describe().transpose().round(2))

    # Note: Binary variables moved to categorical

    with tab2:
        st.markdown("### Overview of Categorical Features")

        # Two columns for station features
        cols1, cols2, cols3 = st.columns(3)
        with cols1:
            unique_start = combined['Start station'].nunique()
            st.metric(label="Start Stations (Unique)", value=f"{unique_start:,}")
        with cols2:
            unique_end = combined['End station'].nunique()
            st.metric(label="End Stations (Unique)", value=f"{unique_end:,}")
        with cols3:
            same_start_end_count = combined['Same Start-End'].sum()
            st.metric(label="Same Start-End Trips", value=f"{same_start_end_count:,}")

        st.divider()
        
        # Distribution of other categorical features
        cols1, cols2 = st.columns(2)
        
        with cols1:
            # Distribution of `is_Weekend`
            value_counts = combined["is_Weekend"].value_counts().reset_index()
            value_counts.columns = ["is_Weekend", 'Count']
            value_counts["is_Weekend"] = value_counts["is_Weekend"].replace({1: 'Weekend', 0: 'Weekday'})
            
            fig = px.pie(
            value_counts,
            names="is_Weekend",
            values='Count',
            title="Distribution of `is_Weekend`",
            color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
        
        with cols2:
            # Distribution of `Time_of_Day`
            value_counts = combined["Time_of_Day"].value_counts().reset_index()
            value_counts.columns = ["Time_of_Day", 'Count']
            
            fig = px.pie(
            value_counts,
            names="Time_of_Day",
            values='Count',
            title="Distribution of `Time_of_Day`",
            color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
        
        st.divider()

        # Distribution of `Weather Description`
        value_counts = combined["Weather Description"].value_counts().reset_index()
        value_counts.columns = ["Weather Description", 'Count']
        
        fig = px.bar(
            value_counts,
            x="Weather Description",
            y='Count',
            title="Distribution of `Weather Description`",
            color='Count',
            color_continuous_scale=px.colors.qualitative.Pastel
        )
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.update_layout(showlegend=False, xaxis_title=None, coloraxis_showscale=False)
        st.plotly_chart(fig)
    
    # Save combined dataset to session state for future use
    # combined.to_csv("combined_data.csv", index=False)
    st.session_state['combined_data'] = combined
    st.success("Combined dataset saved successfully.")

    st.markdown("### Ready to Explore")
    st.write("With clean and merged data, we're set to uncover insights about London's bike-sharing trends!")
