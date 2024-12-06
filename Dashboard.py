import streamlit as st

st.set_page_config(layout="wide", page_title="Ride the Weather: London Bike-Weather Insights")

# Title and Welcome Section
st.markdown("<h1 style='text-align: left; color: black;'>>🚴‍♂️ 🚴‍♂️🚴‍♂️ Ride the Weather</h1>", unsafe_allow_html=True)
st.divider()
st.header("""Welcome to your one-stop destination to explore London's bike-sharing trends! """)

# Key Highlights Section
st.markdown("Here are some exciting stats:")
highlight_col1, highlight_col2, highlight_col3 = st.columns(3)

with highlight_col1:
   st.metric(label=" 🚴 Total Rides", value="776,527", delta="15% ↑ from July")
with highlight_col2:
   st.metric(label="📍 Busiest Station", value="Hyde Park Corner")
with highlight_col3:
   st.metric(label="⏰ Most Active Hour", value="5:00 PM - 6:00 PM")

st.write("This dashboard helps you explore how weather, time, and city life come together to create fascinating biking tpatterns in **August 2023**")
st.divider()

# Sidebar for multipage navigation
intro_section = st.sidebar.radio("Go to", 
                                 ["🌞 Why August?", 
                                  "🚀 Discover What You Can Do Here", 
                                  "🚴‍♂️ Let’s Ride the Weather Together!"])

# Why This App Section
if intro_section == "🌞 Why August?":
   st.header("🌞 Why August?")
   st.markdown("""
   August is all about sunny vibes, busy streets, and lots of biking! It’s when:
   - **The weather is warm**, making it perfect for outdoor rides.
   - **Tourists explore the city**, adding to the bike-sharing buzz.
   - **Locals commute or relax**, using bikes for quick trips or scenic rides.
   """)

# Explore Section
elif intro_section == "🚀 Discover What You Can Do Here":
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

# Dashboard Workflow Overview Section
else:
   st.header("🚴‍♂️ Let’s Ride the Weather Together!")
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