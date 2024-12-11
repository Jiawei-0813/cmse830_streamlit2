import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sequential, qualitative
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score, calinski_harabasz_score

st.set_page_config(layout="wide",
                     page_title="📈 Exploring Bike Trends"
                     )

# --- Load Combined Dataset ---
if 'combined_data' in st.session_state:
    combined = st.session_state['combined_data']
    st.success("Combined dataset loaded successfully from session state.")
else:
    st.error("Combined dataset not found in session state. Please go back to the previous page to process the data.")
    st.stop()

# --- Page Header ---
st.title("📈 Exploratory Bike-Sharing Trends")

# --- Tabs for Analysis Sections ---
tab_names = ["⏰ Peak Times", "🌦 Weather Impact", "🚏 Station Analysis"]
tabs = st.tabs(tab_names)

# --- ⏰ Peak Times ---
with tabs[0]:
    st.subheader("⏰ Peak Times: When do Londoners ride the most?")
    st.markdown("Analyze bike-sharing activity trends based on time of day, day of the month, and weekdays vs weekends.")

    st.divider()

    time_option = st.selectbox(
        "Choose a focus area for peak times analysis:",
        ["Daily Trends (Day of Month)", "Time of Day Patterns", "Weekdays vs Weekends"]
    )

    # Display based on Selection
    if time_option == "Daily Trends (Day of Month)":
        st.subheader("Daily Bike-Sharing Activity")
        st.markdown("Explore how bike-sharing activity changes across different days of the month.")

        # Daily Trends Plot
        daily_counts = combined.groupby("Day_of_Month").size().reset_index(name="Number of Bike-Sharing Trips")
        fig_daily = px.line(
            daily_counts,
            x="Day_of_Month",
            y="Number of Bike-Sharing Trips",
            labels={"Day_of_Month": "Day", "Number of Bike-Sharing Trips": "Number of Bike-Sharing Trips"},
            markers=True,
            line_shape="linear",
        )
        fig_daily.update_traces(mode="lines+markers", line=dict(color="blue"))
        fig_daily.add_scatter(
            x=daily_counts["Day_of_Month"],
            y=daily_counts["Number of Bike-Sharing Trips"],
            fill="tozeroy",
            fillcolor="rgba(173, 216, 230, 0.3)",  # Light blue shading
            line=dict(width=0),
            name="Shading",
            showlegend=False  # Hide the legend for shading
        )
        fig_daily.update_layout(
            hovermode="x unified",
            xaxis=dict(tickmode='linear', dtick=1, range=[1, 31]),  # Ensure all x-axis labels are shown and range is 1-31
            xaxis_title="Day of the Month",
            yaxis_title="Number of Bike-Sharing Trips",
            margin=dict(l=0, r=0, t=30, b=0)  # Adjust margins for better readability
        )

        # Annotate peaks
        peak_days = daily_counts.nlargest(2, "Number of Bike-Sharing Trips")
        for _, peak_day in peak_days.iterrows():
            fig_daily.add_annotation(
            x=peak_day["Day_of_Month"],
            y=peak_day["Number of Bike-Sharing Trips"],
            text=f"Peak: {peak_day['Day_of_Month']}th, {peak_day['Number of Bike-Sharing Trips']} trips",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color="blue"),
            arrowcolor="blue"
            )

        st.plotly_chart(fig_daily)

        # Observations
        st.markdown("""
        **Observations:**
        - Bike-sharing activity shows fluctuations throughout the month.
        """)

    elif time_option == "Time of Day Patterns":
        st.subheader("Bike-Sharing by Time of Day")
        st.markdown("Explore bike-sharing activity across different times of the day (Morning, Afternoon, Evening, and Night).")

        # Time of Day Plot
        time_of_day_counts = combined.groupby(["Day_of_Month", "Time_of_Day"]).size().reset_index(name="Number of Bike-Sharing Trips")
        fig_time_of_day = px.line(
            time_of_day_counts,
            x="Day_of_Month",
            y="Number of Bike-Sharing Trips",
            color="Time_of_Day",
            labels={"Day_of_Month": "Day", "Number of Bike-Sharing Trips": "Number of Bike-Sharing Trips"},
            category_orders={"Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"]},
            markers=True,
            color_discrete_map={
                "Morning": "orange",
                "Afternoon": "green",
                "Evening": "purple",
                "Night": "blue",
            },
        )
        fig_time_of_day.update_layout(hovermode="x unified")
        st.plotly_chart(fig_time_of_day)

        # Observations
        st.markdown("""
        **Observations:**
        - **Morning** and **Evening** exhibit the highest bike-sharing activity, reflecting commuter trends.
        - **Afternoon** activity is moderate, while **Night** sees consistently low activity.
        """)

    elif time_option == "Weekdays vs Weekends":
        st.subheader("Hourly Bike-Sharing: Weekday vs Weekend Activity")
        st.markdown("Analyze how bike-sharing activity differs between weekdays and weekends at various times of the day.")

        # Group data for weekday and weekend
        weekday_counts = combined[combined["is_Weekend"] == 0].groupby("Hour_of_Day").size().reset_index(name="Weekday Count")
        weekend_counts = combined[combined["is_Weekend"] == 1].groupby("Hour_of_Day").size().reset_index(name="Weekend Count")

        # Plot weekday vs weekend activity
        fig_weekday_vs_weekend = px.line(
            x=weekday_counts["Hour_of_Day"],
            y=weekday_counts["Weekday Count"],
            labels={"x": "Hour of Day", "y": "Number of Bike-Sharing"}
        )
        fig_weekday_vs_weekend.add_scatter(
            x=weekend_counts["Hour_of_Day"],
            y=weekend_counts["Weekend Count"],
            mode="lines+markers",
            line=dict(color="orange"),
            name="Weekend"
        )
        fig_weekday_vs_weekend.update_traces(
            mode="lines+markers",
            marker=dict(size=8),
        )
        fig_weekday_vs_weekend.update_layout(
            showlegend=False,
            xaxis=dict(title="Hour of Day", tickmode='linear', dtick=1, range=[0, 24]),  # Ensure all x-axis labels are shown
            yaxis=dict(title="Number of Bike-Sharing Trips", range=[0, max(weekday_counts["Weekday Count"].max(), weekend_counts["Weekend Count"].max())])
        )

        # Find and label peaks
        weekday_peak = weekday_counts.loc[weekday_counts["Weekday Count"].idxmax()]
        weekend_peak = weekend_counts.loc[weekend_counts["Weekend Count"].idxmax()]

        fig_weekday_vs_weekend.add_annotation(
            x=weekday_peak["Hour_of_Day"],
            y=weekday_peak["Weekday Count"],
            text=f"Weekday Peak: {weekday_peak['Hour_of_Day']}h, {weekday_peak['Weekday Count']} trips",
            showarrow=True,
            arrowhead=2,
            font=dict(color="blue"),
            arrowcolor="blue",
            ax=0,
            ay=-40
        )
        fig_weekday_vs_weekend.add_annotation(
            x=weekend_peak["Hour_of_Day"],
            y=weekend_peak["Weekend Count"],
            text=f"Weekend Peak: {weekend_peak['Hour_of_Day']}h, {weekend_peak['Weekend Count']} trips",
            showarrow=True,
            arrowhead=2,
            font=dict(color="orange"),
            arrowcolor="orange",
            ax=0,
            ay=-40
        )

        st.plotly_chart(fig_weekday_vs_weekend)

        # Observations
        st.markdown("""
        **Observations:**
        - **Weekdays** exhibit two prominent peaks: morning (8–9 AM) and evening (5–7 PM), likely due to commuting patterns.
        - **Weekends** show a more consistent activity level throughout the day, with a slight increase in the afternoon.
        - Both weekdays and weekends have minimal activity during late-night hours (midnight to 5 AM).
        """)
        
# --- 🌦 Weather Impact ---
with tabs[1]:
    st.subheader("🌦Weather Impact")
    st.markdown("Analyze how various weather conditions and meteorological factors associate bike-sharing activity.")

    st.divider()

    weather_option = st.selectbox(
        "Choose a focus area for weather impact analysis:",
        ["Weather Conditions", "Rainy vs Not Rainy", "Meteorological Factors"]
    )

    # Display based on Selection
    if weather_option == "Weather Conditions":
        st.subheader("Bike-Sharing by Weather Conditions")
        st.markdown("Explore how different weather conditions impact bike-sharing trends.")

        # Aggregate data by weather description
        weather_counts = combined.groupby("Weather Description").size().reset_index(name="Bike-Sharing Trips")
        weather_counts["Percentage"] = weather_counts["Bike-Sharing Trips"] / weather_counts["Bike-Sharing Trips"].sum() * 100
        weather_counts = weather_counts.sort_values(by="Bike-Sharing Trips", ascending=False)

        # Bar plot
        fig_weather_cond = px.bar(
            weather_counts,
            x="Weather Description",
            y="Bike-Sharing Trips",
            color="Bike-Sharing Trips",
            color_continuous_scale="Viridis",
            text="Percentage",
            labels={"Bike-Sharing Trips": "Number of Bike-Sharing Trips", "Percentage": "Percentage (%)"},
        )
        fig_weather_cond.update_layout(
            xaxis_title=None,  # Remove x-axis title
            showlegend=False,  # Hide the legend
            coloraxis_showscale=False  # Hide color legend
        )
        fig_weather_cond.update_traces(
            texttemplate="%{y:,} trips<br>%{text:.1f}%",
            textposition="outside",
        )
        st.plotly_chart(fig_weather_cond)
        
        # Observations
        st.markdown("""
        **Observations:**
        - **Overcast** and **Partly Cloudy** conditions see the highest bike-sharing activity.
        - **Rain**-related conditions lead to fewer trips, likely due to discomfort and safety concerns.
        """)

    elif weather_option == "Rainy vs Not Rainy":
        st.subheader("Daily Bike-Sharing: Rainy vs Not Rainy")
        st.markdown("Compare bike-sharing activity between rainy and non-rainy days over the course of the month.")

        # Aggregate data for rainy vs not rainy
        rainy_counts = combined.groupby(["Date", "is_Rainy"]).size().reset_index(name="Bike-Sharing Trips")
        rainy_counts["Rain Condition"] = rainy_counts["is_Rainy"].map({0: "Not Rainy", 1: "Rainy"})

        # Rolling window slider
        rolling_window = st.slider("Select Rolling Window Size (Days):", 1, 7, 3)

        # Add rolling average lines for clearer trends
        rainy_counts_rolling = rainy_counts.copy()
        rainy_counts_rolling["Rolling Average"] = rainy_counts.groupby("Rain Condition")["Bike-Sharing Trips"].transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
        fig_smoothed_rain = px.line(
            rainy_counts_rolling,
            x="Date",
            y="Rolling Average",
            color="Rain Condition",
            color_discrete_map={"Not Rainy": "lightgreen", "Rainy": "darkgreen"},
            title="Smoothed Daily Bike-Sharing: Rainy vs Not Rainy",
            labels={"Date": "Date", "Rolling Average": f"Rolling Average ({rolling_window} Days)", "Rain Condition": "Condition"},
        )
        fig_smoothed_rain.update_layout(
            hovermode="x unified",
            xaxis=dict(tickformat="%b %d", title=None),  # Remove x-axis title
            yaxis=dict(title="Rolling Average (Bike-Sharing Trips)"),
            legend=dict(x=0.8, y=1, title="Rain Condition"),
        )

        st.plotly_chart(fig_smoothed_rain)

        # Observations
        st.markdown(f"""
        **Observations:**
        - **Rainy days** show consistently lower bike-sharing activity compared to non-rainy days, with minimal variation.
        - The rolling averages emphasize a persistent gap in bike-sharing trends, with rain acting as a strong deterrent for users across the entire period.
        """)
                    
        st.divider()

    # --- Impact of Meteorological Factors ---
    elif weather_option == "Meteorological Factors":
        
        # Violin Plot: Distribution of Bike-Sharing by Meteorological Metrics
        st.subheader("Distribution Across Meteorological Metrics")
        weather_cols = [
            "temperature_2m_scaled", "apparent_temperature_scaled",
            "wind_speed_10m_scaled", "relative_humidity_2m_scaled"
        ]
        weather_labels = {
            "temperature_2m_scaled": "Temperature",
            "apparent_temperature_scaled": "Apparent Temperature",
            "wind_speed_10m_scaled": "Wind Speed",
            "relative_humidity_2m_scaled": "Humidity"
        }
        weather_long = pd.melt(
            combined, value_vars=weather_cols,
            var_name="Meteorological Metric", value_name="Value"
        )
        weather_long["Meteorological Metric"] = weather_long["Meteorological Metric"].map(weather_labels)

        fig_violin_weather = px.violin(
            weather_long,
            x="Meteorological Metric",
            y="Value",
            color="Meteorological Metric",
            box=True,
            color_discrete_sequence=["#FFA07A", "#FA8072", "#6495ED", "#98FB98"],  # Consistent coloring
        )
        fig_violin_weather.update_layout(
            xaxis_title=None,
            yaxis_title="Value (Scaled)",
            showlegend=False
        )
        st.plotly_chart(fig_violin_weather)

        st.divider()

        # Correlation Heatmap for Meteorological Metrics
        st.subheader("Correlation Between Meteorological Metrics")
        numeric_weather = combined[[
            "temperature_2m_scaled", "apparent_temperature_scaled",
            "wind_speed_10m_scaled", "relative_humidity_2m_scaled"
        ]]
        numeric_weather.columns = ["Temperature", "Apparent Temperature", "Wind Speed", "Humidity"]
        corr_matrix = numeric_weather.corr()
        fig_corr_weather = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="Viridis",
        )
        fig_corr_weather.update_layout(
            height=500,
            xaxis_title=None,  # Hide x-axis title
            yaxis_title=None,  # Hide y-axis title
            showlegend=False   # Hide legend
        )
        st.plotly_chart(fig_corr_weather)

        # Observations
        st.markdown("""
        **Observations:**
        - **Temperature** and **Apparent Temperature** show strong positive correlations, suggesting they similarly influence bike-sharing activity.
        - **Humidity** has a weak negative correlation, indicating a slight deterrent effect on bike-sharing.
        - **Wind Speed** exhibits minimal correlation, showing limited impact on bike-sharing trends.
        - The correlation between **Humidity** and **Temperature** differs from that with **Apparent Temperature**, hinting at different weather patterns at play.
        """)

# --- 🚏 Station Analysis ---
with tabs[2]:
    st.subheader("🚏Station Hotspots")
    st.markdown("Analyze bike-sharing trends based on stations.")

    st.divider()

    station_analysis = st.selectbox(
        "Choose a station-related question to explore:",
        [
            "Busiest Stations",
            "Clustering"
        ],
    )

    if station_analysis == "Busiest Stations":
        st.subheader("Which stations are the busiest?")
        station_option = st.radio(
            "Select station type for analysis:",
            ["Start station", "End station"]
        )
        top_n = st.slider(f"Select top **{station_option}s**:", min_value=1, max_value=50, value=10)

        # Generate data for busiest stations
        station_counts = combined[station_option].value_counts().head(top_n)
        title = f"Busiest {top_n} {station_option}s"

        # Create DataFrame for plotting
        station_df = station_counts.reset_index()
        station_df.columns = ["Station Name", "Count"]

        # Create Bar Plot using Plotly
        fig_busiest = px.bar(
            station_df,
            x="Count",
            y="Station Name",
            orientation="h",
            title=title,
            labels={"Count": "Number of Bike-Sharing Trips", "Station Name": f"{station_option} Name"},
            color="Station Name",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text="Count"  # Add text to display the count near the bar
        )
        fig_busiest.update_layout(showlegend=False, xaxis=dict(showticklabels=False))
        fig_busiest.update_traces(textposition='outside')  # Position the text outside the bars
        st.plotly_chart(fig_busiest)

        # Calculate and display Percentage Coverage
        if 'station_df' in locals() and not station_df.empty:  # Check if station_df exists and is not empty
            percentage = (station_df["Count"].sum() / len(combined)) * 100
            st.markdown(f"Top {top_n} **{station_option.lower()}** stations account for **{percentage:.2f}%** of total trips.")
        else:
            st.markdown("No data available for the selected station option.")

    elif station_analysis == "Clustering":
        st.markdown("Group stations based on the number of trips using K-Means clustering.")

        # Step 1: Select Number of Clusters
        n_clusters = st.slider("Choose the number of clusters (k):", min_value=2, max_value=10, value=4, step=1)

        # Step 2: Prepare Data
        station_data = combined[["Start station", "End station"]].copy()
        station_data["Station"] = station_data["Start station"].combine_first(station_data["End station"])
        station_data_counts = station_data["Station"].value_counts().reset_index()
        station_data_counts.columns = ["Station", "Number of Bike-Sharing Trips"]

        # Check if data is empty
        if station_data_counts.empty:
            st.error("No data available for clustering. Please check the dataset.")
            st.stop()

        # Standardize data
        scaler = StandardScaler()
        station_data_counts["Trips_Scaled"] = scaler.fit_transform(station_data_counts[["Number of Bike-Sharing Trips"]])

        # Step 3: Apply K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        station_data_counts["Cluster"] = kmeans.fit_predict(station_data_counts[["Trips_Scaled"]])

        # Step 4: Cluster Summary
        st.subheader("Explore Cluster Summary")
        st.markdown(
            "Summary of clusters with the number of stations, average trips, standard deviation, "
            "and example stations."
        )
        cluster_summary = station_data_counts.groupby("Cluster").agg(
            Cluster_Size=("Station", "count"),
            Avg_Trips=("Number of Bike-Sharing Trips", "mean"),
            Std_Trips=("Number of Bike-Sharing Trips", "std"),
            Example_Stations=("Station", lambda x: ", ".join(x[:3])),
        ).reset_index()
        st.dataframe(cluster_summary)


        # Automatically generate insights for each cluster
        insights = [
            f"- **Cluster {row['Cluster']}**: {row['Cluster_Size']} stations with an average of "
            f"{row['Avg_Trips']:.2f} trips per station (std. dev. {row['Std_Trips']:.2f}). "
            for _, row in cluster_summary.iterrows()
        ]
        st.markdown("\n".join(insights))

        st.divider()

        # Step 5: Visualize Clusters
        st.subheader("Visualize Clusters")

        # Bar Chart: Clustered Station Usage
        fig_cluster = px.bar(
            station_data_counts,
            x="Station",
            y="Number of Bike-Sharing Trips",
            color="Cluster",
            title="Clustered Station Usage",
            labels={"Station": "Station Name", "Number of Bike-Sharing Trips": "Bike Usage Count"},
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_cluster.update_layout(xaxis=dict(showticklabels=False), yaxis_title="Bike Usage Count")
        st.plotly_chart(fig_cluster, use_container_width=True, key="clustered_station_usage")

        # Automatically highlight the highest and lowest usage clusters
        highest_usage_cluster = cluster_summary.loc[cluster_summary["Avg_Trips"].idxmax()]
        lowest_usage_cluster = cluster_summary.loc[cluster_summary["Avg_Trips"].idxmin()]
        st.markdown(
            f"- **Cluster {highest_usage_cluster['Cluster']}** has the highest average trips "
            f"({highest_usage_cluster['Avg_Trips']:.2f}), representing the most active stations."
        )
        st.markdown(
            f"- **Cluster {lowest_usage_cluster['Cluster']}** has the lowest average trips "
            f"({lowest_usage_cluster['Avg_Trips']:.2f}), representing low-demand stations."
        )

        # Heatmap: Cluster Feature Comparison
        heatmap_data = station_data_counts.groupby("Cluster")[["Number of Bike-Sharing Trips", "Trips_Scaled"]].mean().reset_index()
        heatmap_fig = px.imshow(
            heatmap_data.set_index("Cluster").T,
            color_continuous_scale="Viridis",
            title="Cluster Feature Comparison",
            labels={"x": "Cluster", "y": "Feature", "color": "Value"},
            text_auto=True,
        )
        st.plotly_chart(heatmap_fig, use_container_width=True, key="cluster_feature_comparison")

        for feature in heatmap_data.columns[1:]:  # Skip Cluster column
            max_cluster = heatmap_data.loc[heatmap_data[feature].idxmax(), "Cluster"]
            min_cluster = heatmap_data.loc[heatmap_data[feature].idxmin(), "Cluster"]
            st.markdown(
            f"- **{feature}:** Cluster {max_cluster} has the highest value, while Cluster {min_cluster} has the lowest."
            )

        st.divider()

        # Step 6: Evaluate Clusters
        st.subheader("Evaluate Clustering Metrics")

        # Compute metrics for different values of k
        metrics = {
            "Number of Clusters (k)": [],
            "Distortion (Inertia)": [],
            "Silhouette Score": [],
            "Calinski-Harabasz Index": [],
        }

        for k in range(2, 11):
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            kmeans_temp.fit(station_data_counts[["Trips_Scaled"]])
            metrics["Number of Clusters (k)"].append(k)
            metrics["Distortion (Inertia)"].append(kmeans_temp.inertia_)
            metrics["Silhouette Score"].append(
            silhouette_score(station_data_counts[["Trips_Scaled"]], kmeans_temp.labels_)
            )
            metrics["Calinski-Harabasz Index"].append(
            calinski_harabasz_score(station_data_counts[["Trips_Scaled"]], kmeans_temp.labels_)
            )

        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df)

        optimal_k = metrics_df.loc[metrics_df["Silhouette Score"].idxmax(), "Number of Clusters (k)"]
        st.markdown(
            f"- **Optimal k:** Based on the highest Silhouette Score, the optimal number of clusters is **{optimal_k}**."
        )

        # Plot Metrics Comparison
        fig_metrics = px.line(
            metrics_df,
            x="Number of Clusters (k)",
            y=["Distortion (Inertia)", "Silhouette Score", "Calinski-Harabasz Index"],
            title="Clustering Metrics Comparison for Different k",
            labels={"value": "Metric Value", "variable": "Metric", "Number of Clusters (k)": "Number of Clusters (k)"},
        )
        st.plotly_chart(fig_metrics, use_container_width=True, key="metrics_comparison")
