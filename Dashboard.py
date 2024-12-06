import streamlit as st

st.set_page_config(layout="wide")

st.title("Ride the Weather: London Bike-Weather Insights")

# Introduction
st.markdown("""
Welcome to **Ride the Weather**, your ultimate guide to London's bike-sharing ecosystem.
            """)

# Why This App Section
st.header("🌟 Why August?")
st.markdown("""
August is all about sunny vibes, busy streets, and lots of biking! It’s when:
- **The weather is warm**, making it perfect for outdoor rides.
- **Tourists explore the city**, adding to the bike-sharing buzz.
- **Locals commute or relax**, using bikes for quick trips or scenic rides.

This dashboard helps you explore how bikes, weather, and time shape the rhythms of London life!
""")

# Key Highlights Section
st.subheader("📊 Highlights from August")
st.markdown("""
Here are some fun stats from August 2023:
- **Total Rides**: A whopping 776,527 trips!
- **Busiest Station**: Hyde Park Corner, where riders come and go all day.
- **Most Active Hour**: 5:00 PM - 6:00 PM (evening rush, anyone?).
""")
st.metric(label="Total Rides", value="776,527", delta="15% ↑ compared to July")
st.metric(label="Busiest Station", value="Hyde Park Corner")
st.metric(label="Most Active Hour", value="5:00 PM - 6:00 PM")

# Explore Section
st.header("🚀 What Can You Do Here?")
st.markdown("""
Bike-sharing is transforming how cities move, and London is no exception. It’s fast, affordable, and eco-friendly—perfect for navigating the city or soaking up its iconic sights. 

More than just a way to get from point A to point B, London’s bike-sharing is a culture, a movement, and a story. From early-morning commutes to leisurely weekend rides through Hyde Park, these bikes connect people to the city in a way that’s both practical and personal.

Every ride brings people closer—to each other, to landmarks, and to experiences that only two wheels can offer.

There’s a lot to explore:
1. **Bike Trends**: See how the weather affected rides—sunny days, rainy mornings, and everything in between.
2. **Peak Times**: Find out when Londoners love to ride most.
3. **Station Hotspots**: Check out where rides begin and end across the city.

This dashboard is here to make bike-sharing fun and easy to understand!
""")

# App Workflow Overview
st.header("🛠 Dashboard Workflow Overview")
st.markdown(
    """
    1. **Data Preprocessing**:
       - Clean and preprocess raw data from bike-sharing and weather datasets.
       - Merge datasets on a common `Date` column for seamless analysis.
    2. **Exploratory Data Analysis (EDA)**:
       - Visualize demand patterns based on time, weather, and location.
       - Investigate feature relationships and distributions.
    3. **Feature Engineering**:
       - Extract features like day of the week, time of day, and weather categories.
       - Encode categorical variables and scale numerical variables.
    4. **Modeling and Predictions**:
       - Build regression and ensemble models to predict bike-sharing demand.
       - Evaluate model performance and visualize actual vs. predicted values.
    """
)

# Footer
st.markdown("**Let’s Ride the Weather together!** 🚴‍♂️")