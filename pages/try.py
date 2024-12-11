import streamlit as st
import plotly.express as px
import pandas as pd

# Assuming `combined` is a DataFrame that you have already loaded
combined = pd.DataFrame()  # Replace with actual data loading code

# Define tabs
tabs = st.tabs(["⏰ Peak Times", "🌦 Weather Impact", "📍 Station Analysis"])

# --- ⏰ PEAK TIMES TAB ---
with tabs[0]:
    st.header("⏰ Peak Times")
    st.markdown("Analyze bike-sharing trends over time, including daily trends, time of day patterns, and weekday vs weekend activity.")

    # 1. Daily Trends
    st.subheader("Daily Bike-Sharing Activity")
    daily_counts = combined.groupby("Day_of_Month").size().reset_index(name="Number of Bike-Sharing Trips") # type: ignore
    fig_daily = px.line(
        daily_counts,
        x="Day_of_Month",
        y="Number of Bike-Sharing Trips",
        title="Daily Bike-Sharing Activity",
        labels={"Day_of_Month": "Day of Month", "Number of Bike-Sharing Trips": "Count"},
        markers=True,
        line_shape="linear",
        color_discrete_sequence=["#007BFF"],
    )
    fig_daily.update_traces(fill="tozeroy", fillcolor="rgba(0, 123, 255, 0.2)")
    st.plotly_chart(fig_daily, key="daily_trends")

    # 2. Time of Day Patterns
    st.subheader("Bike-Sharing by Time of Day")
    time_of_day_counts = combined.groupby(["Day_of_Month", "Time_of_Day"]).size().reset_index(name="Number of Bike-Sharing Trips")
    fig_time_of_day = px.line(
        time_of_day_counts,
        x="Day_of_Month",
        y="Number of Bike-Sharing Trips",
        color="Time_of_Day",
        title="Bike-Sharing by Time of Day and Day of Month",
        labels={"Day_of_Month": "Day of Month", "Number of Bike-Sharing Trips": "Count"},
        category_orders={"Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"]},
        color_discrete_map={
            "Morning": "orange",
            "Afternoon": "green",
            "Evening": "purple",
            "Night": "blue",
        },
        markers=True,
    )
    st.plotly_chart(fig_time_of_day, key="time_of_day_patterns")

    # 3. Weekday vs Weekend
    st.subheader("Hourly Bike-Sharing: Weekday vs Weekend")
    hourly_data = combined.groupby(["Hour_of_Day", "is_Weekend"]).size().reset_index(name="Count")
    hourly_data["Type"] = hourly_data["is_Weekend"].map({0: "Weekday", 1: "Weekend"})
    fig_weekday_vs_weekend = px.line(
        hourly_data,
        x="Hour_of_Day",
        y="Count",
        color="Type",
        title="Hourly Bike-Sharing: Weekday vs Weekend",
        labels={"Hour_of_Day": "Hour of Day", "Count": "Count"},
        color_discrete_map={"Weekday": "blue", "Weekend": "orange"},
        markers=True,
    )
    st.plotly_chart(fig_weekday_vs_weekend, key="weekday_vs_weekend")

# --- 🌦 WEATHER IMPACT TAB ---
with tabs[1]:
    st.header("🌦 Weather Impact")
    st.markdown("Explore how weather conditions and meteorological factors influence bike-sharing trends.")

    # 1. Weather Description
    st.subheader("Bike-Sharing by Weather Description")
    weather_counts = combined.groupby("Weather Description").size().reset_index(name="Bike-Sharing Trips")
    fig_weather_desc = px.bar(
        weather_counts,
        x="Weather Description",
        y="Bike-Sharing Trips",
        title="Bike-Sharing by Weather Description",
        labels={"Bike-Sharing Trips": "Count"},
        color_discrete_sequence=px.colors.sequential.Greens,
    )
    st.plotly_chart(fig_weather_desc, key="weather_description")

    # 2. Rainy vs Not Rainy
    st.subheader("Rainy vs Not Rainy Days")
    rain_data = combined.groupby(["is_Rainy", "Hour_of_Day"]).size().reset_index(name="Bike-Sharing Trips")
    rain_data["Rain Status"] = rain_data["is_Rainy"].map({0: "Not Rainy", 1: "Rainy"})
    fig_rain = px.line(
        rain_data,
        x="Hour_of_Day",
        y="Bike-Sharing Trips",
        color="Rain Status",
        title="Bike-Sharing on Rainy vs Not Rainy Days",
        labels={"Hour_of_Day": "Hour of Day", "Bike-Sharing Trips": "Count"},
        color_discrete_map={"Not Rainy": "lightgreen", "Rainy": "darkgreen"},
        markers=True,
    )
    st.plotly_chart(fig_rain, key="rainy_vs_not_rainy")

    # 3. Numeric Weather Factors
    st.subheader("Impact of Temperature, Humidity, and Wind Speed")
    fig_numeric_weather = px.scatter(
        combined,
        x="temperature_2m_scaled",
        y="Total duration (m)",
        color="is_Rainy",
        trendline="ols",
        labels={"temperature_2m_scaled": "Temperature (Scaled)", "Total duration (m)": "Count"},
        color_discrete_map={0: "lightgreen", 1: "darkgreen"},
        title="Bike-Sharing vs Temperature (Rainy vs Not Rainy)",
    )
    st.plotly_chart(fig_numeric_weather, key="numeric_weather")

# --- 📍 STATION ANALYSIS TAB ---
with tabs[2]:
    st.header("📍 Station Analysis")
    st.markdown("Explore station-specific trends, including the busiest stations and clustering analysis.")

    # Analysis of busiest stations
    st.subheader("Busiest Stations")
    top_stations = combined["Start station"].value_counts().reset_index(name="Count").head(10)
    fig_busiest = px.bar(
        top_stations,
        x="Count",
        y="index",
        orientation="h",
        title="Top 10 Busiest Start Stations",
        labels={"Count": "Number of Bike-Sharing Trips", "index": "Station Name"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_busiest, key="busiest_stations")