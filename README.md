# 🎈 Ride the Weather 

[[!Open in Streamlit] (https://cmse830app2-ridetheweather.streamlit.app/)]

Haver you ever wondered what drives bike-sharing in London? 

This Streamlit dashboard utilizes two publicly available datasets:

- **London Bike-Share Usage Dataset**: This dataset contains bike-sharing trips recorded by Transport for London (TfL) during August 2023, including the start and end points of each trip, the bike used, and the duration of the ride.
- **Open-Meteo Weather Data**: Weather data for the same period, sourced from the Open-Meteo API, detailing the start and end points of each trip, the bike used, and the duration of the ride.

## 🚴‍♂️ Let’s Ride the Weather Together!
Our dashboard is structured to make your exploration smooth and insightful:

Data Preprocessing:
- Clean and preprocess raw data from bike-sharing and weather datasets.
- Combine datasets on a common Date column for seamless analysis.

Exploratory Data Analysis (EDA):
- Visualize trends by time, weather, and location.
- Explore demand patterns and feature relationships.

Feature Engineering:
- Extract and transform features like day of the week, time of day, and weather categories.
- Encode categorical variables and scale numerical features.

Modeling and Predictions:
- Build models to predict bike-sharing demand.
- Evaluate model performance with interactive visualizations.

### From raw data to actionable insights, this app takes you through every step!




### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run Dashboard.py
   ```

