import streamlit as st

st.set_page_config(layout="wide", page_title="Ride the Weather: London Bike-Weather Insights")

# --- Title and Welcome Section ---
col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.image("https://media0.giphy.com/media/20NUvxKjMv5YkMykZS/giphy.gif?cid=ecf05e470hfz4hijiz5cjjrlbrgk6whdnsvtn7kn6z90vzms&ep=v1_stickers_search&rid=giphy.gif&ct=s")

with col2:
    st.markdown("<h1 style='text-align: left; color: #1E90FF;'> Ride the Weather</h1>", unsafe_allow_html=True)
    st.markdown("<div style='display: flex; align-items: center; height: 100%;'><h3 style='color: #333333;'>Welcome to your one-stop destination to explore London's bike-sharing trends!</h3></div>", unsafe_allow_html=True)

st.divider()

# --- Key Highlights Section ---
st.markdown("<h3 style='font-size: 20px; color: #FF4500;'>Do you know in London August 2023:</h3>", unsafe_allow_html=True)
highlight_col1, highlight_col2, highlight_col3 = st.columns(3)

with highlight_col1:
    st.metric(label=" **🚴 Total Rides**", value="776,527")
with highlight_col2:
    st.metric(label="📍 **Busiest Station**", value="Hyde Park Corner")
with highlight_col3:
    st.metric(label="⏰ **Most Active Hour**", value="5PM - 7PM")

st.divider()

# --- Sidebar for multipage navigation ---
intro_section = st.sidebar.radio("Go to", 
                                 ["🚴‍♂️ Why Bike-sharing Analysis?", 
                                  "🚴‍♂️ Let’s Ride the Weather Together!"])

# --- Why This App Section ---
if intro_section == "🚴‍♂️ Why Bike-sharing Analysis?":
    st.header("🚴‍♂️ Why London Bike-sharing Analysis?")
    st.markdown("""
    Bike-sharing is transforming how cities move, and London is no exception. It’s fast, affordable, and eco-friendly—
    a perfect solution for both locals and visitors. Here’s why bike-sharing analysis is exciting:
    - **Sustainable Transport**: Reduces traffic congestion and carbon emissions.
    - **Urban Mobility**: Provides a flexible and convenient way to navigate the city.
    - **Data-Driven Insights**: Helps cities optimize bike-sharing systems and infrastructure.
    - **Community Engagement**: Encourages active lifestyles and social connections.

    London’s bike-sharing isn’t just about getting around—it’s about connecting people, places, and experiences. 
    Whether you’re zipping through traffic, enjoying a park ride, or exploring the city’s landmarks, bikes are part of the rhythm of London life.

    This dashboard is here to make bike-sharing fun and easy to explore! Here’s what you can explore:
    1. **🚲 Ride Counts**: How many bike-sharing rides happened across the city?
    2. **⏰ Peak Times**: When do Londoners ride the most? Morning rush hours or evening wind-downs?
    3. **🌦 Weather Impact**: How do sunny skies or sudden rains influence bike-sharing rides?
    4. **🚏 Station Hotspots**: Which stations are the busiest for starts and stops, and why?

    We’ve made it simple and interactive, so you can explore the data, spot patterns, and maybe even plan your next ride!
    """)

# --- Dashboard Workflow Overview Section ---
else:
    st.header("🚴‍♂️ Let’s Ride the Weather Together!")
    st.markdown("""
    Our dashboard is structured to make your exploration smooth and insightful:
    1. **Data Preprocessing**:
        - Clean and preprocess raw data from bike-sharing and weather datasets.
        - Combine datasets on a common `Date` column for seamless analysis.
    2. **Exploratory Data Analysis (EDA)**:
        - Visualize trends by time, weather, and stations.
        - Explore demand patterns and feature relationships.
    3. **Feature Engineering**:
        - Extract and transform features like day of the week, time of day, and weather categories.
        - Encode categorical variables and scale numerical features.
    4. **Modeling and Predictions**:
        - Build models to predict bike-sharing demand.
        - Evaluate model performance with interactive visualizations.

    From raw data to actionable insights, this app takes you through every step!
    """)

# --- Break for Visual Section ---
st.divider()