import streamlit as st

st.set_page_config(layout="wide", page_title="Ride the Weather: London Bike-Weather Insights")

# Title and Welcome Section
with st.container():
   st.title("🚴‍♂️ Ride the Weather: London Bike Insights")
   st.markdown("""
   Welcome to **Ride the Weather**, your one-stop destination to explore London's bike-sharing trends! 
   Discover how weather, time, and city life come together to create fascinating biking patterns in **August 2023**.
   """)

st.divider()

# Why This App Section
with st.expander("🌞 Why August?"):
   st.markdown("""
   August is all about sunny vibes, busy streets, and lots of biking! It’s when:
   - **The weather is warm**, making it perfect for outdoor rides.
   - **Tourists explore the city**, adding to the bike-sharing buzz.
   - **Locals commute or relax**, using bikes for quick trips or scenic rides.

   This dashboard helps you explore how bikes, weather, and time shape the rhythms of London life!
   """)

# Key Highlights Section
st.subheader("📊 Highlights from August")
st.markdown("Here are some exciting stats:")
highlight_col1, highlight_col2, highlight_col3 = st.columns(3)

with highlight_col1:
    st.metric(label="Total Rides", value="776,527", delta="15% ↑ from July")
with highlight_col2:
    st.metric(label="Busiest Station", value="Hyde Park Corner")
with highlight_col3:
    st.metric(label="Most Active Hour", value="5:00 PM - 6:00 PM")

st.markdown("""
Londoners and visitors alike hit the pedals, contributing to this vibrant biking ecosystem!
""")

# Explore Section
st.header("🚀 Discover What You Can Do Here")
st.markdown("""
London’s bike-sharing isn’t just about getting around—it’s about connecting people, places, and experiences. 
Whether you’re zipping through traffic, enjoying a park ride, or exploring the city’s landmarks, bikes are part of the rhythm of London life.

This dashboard is here to make bike-sharing fun and easy to understand! Here’s what you can explore:
1. **📅Peak Riding Times**: When do Londoners hit the pedals the most? Morning rush hours or evening wind-downs?
2. **🌦Weather’s Role**: How do sunny skies or sudden showers change riding patterns?
3. **🚏Station Hotspots**: Which stations see the most action, and why?

We’ve made it simple and interactive, so you can explore the data, uncover trends, and maybe even plan your next ride!
""")

st.divider()

# Dashboard Workflow Overview Section
st.header("🛠 Dashboard Workflow Overview")
st.markdown("""
Our dashboard is structured to make your exploration smooth and insightful:
1. **Data Preprocessing**:
   - Clean and preprocess raw data from bike-sharing and weather datasets.
   - Combine datasets on a common `Date` column for seamless analysis.
2. **Exploratory Data Analysis (EDA)**:
   - Visualize trends by time, weather, and location.
   - Explore demand patterns and feature relationships.
3. **Feature Engineering**:
   - Extract and transform features like day of the week, time of day, and weather categories.
   - Encode categorical variables and scale numerical features.
4. **Modeling and Predictions**:
   - Build models to predict bike-sharing demand.
   - Evaluate model performance with interactive visualizations.

From raw data to actionable insights, this app takes you through every step!
""")

# Footer
st.markdown("**Let’s Ride the Weather together!** 🚴‍♂️")