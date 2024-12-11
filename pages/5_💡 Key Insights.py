import streamlit as st

# Sidebar for navigation
summary_section = st.sidebar.radio(
    "Select a section to view:",
    ("Key Insights", "Real-World Implications", "Limitations and Further Steps")
)

# Display content based on sidebar selection
if summary_section == "Key Insights":
    st.markdown("## Key Insights")

    st.markdown("""
    The **Ride the Weather Dashboard** provides valuable insights that help you understand bike usage patterns in London. Here's a summary of the key findings:
    """)

    st.divider()
    
    st.markdown("""
    ### 1. Peak Usage Times
    - **Observation**: Bike-sharing activity is highest during **weekday mornings and evenings**, driven by commuter traffic.
    - **What it means**: These time slots are crucial for ensuring bikes are available where demand is highest. Understanding this can help optimize **station placement** and **bike availability** during peak hours.

    ### 2. Weather Effects on Usage
    - **Observation**: **Mild weather** leads to increased bike-sharing trips, while **rain and high humidity** significantly reduce usage.
    - **What it means**: Weather plays a big role in bike usage patterns. Understanding this can help predict **bike demand** and plan for **redistribution** based on weather forecasts.

    ### 3. Weekdays vs Weekends
    - **Observation**: **Weekdays** show commuter-driven peaks, while **weekends** exhibit more steady usage throughout the day, driven by leisure riders.
    - **What it means**: This suggests that service needs to be **adapted** for weekdays and weekends separately, with **more bikes in commuter-heavy areas** during weekdays and **greater flexibility** for weekend users.

    ### 4. Station-Specific Trends
    - **Observation**: Certain stations show **higher activity**, likely due to their proximity to high-traffic areas like transportation hubs.
    - **What it means**: Recognizing these patterns allows for **focused investments** in high-traffic stations, where bikes are needed most.

    These insights can help inform decisions on how to better serve the community, ensuring **more efficient operations** and **better service delivery**.
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .stMarkdown h3 {
            color: #1E90FF;
        }
        .stMarkdown h2 {
            color: #FF4500;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif summary_section == "Real-World Implications":
    st.markdown("## Real-World Implications")

    st.divider()
    
    st.markdown("""
    Building on the insights provided, here are the **actionable steps** that can be taken to improve the bike-sharing system:

    ### 1. Optimize Station Placement and Bike Allocation
    - Focus on **high-traffic commuter areas**, such as **train stations**, **business districts**, and **transport hubs**, to ensure bikes are available during peak commuter times.
    - Use **real-time data** from the dashboard to **redistribute bikes** and ensure they’re in the right places when demand is highest.

    ### 2. Implement Weather-Based Operations
    - Leverage **weather forecasts** from the dashboard to predict demand and adjust bike availability. For example, increase the number of bikes in popular areas on **sunny days** and **reduce** bikes in **low-demand areas** during rainy weather.
    - Create a **dynamic bike redistribution strategy** that adapts to the weather, improving **efficiency** and ensuring bikes are available when the weather encourages more riders.

    ### 3. Adapt Service for Weekdays and Weekends
    - During **weekdays**, focus on **redistributing bikes** to **commuter-heavy areas**. Ensure bikes are available during rush hours (morning and evening).
    - On **weekends**, increase bike availability in **residential areas** and **leisure zones** like parks or tourist spots, where steady usage is observed throughout the day.

    ### 4. Expand the Fleet Based on Usage Patterns
    - Use the dashboard’s **station-specific data** to identify high-usage areas and **expand** those stations or areas with **additional bikes**.
    - Focus on areas where demand consistently outpaces supply, ensuring bikes are **reallocated or added** to better meet the needs of the riders.
    """)
    st.markdown(
        """
        <style>
        .stMarkdown h3 {
            color: #1E90FF;
        }
        .stMarkdown h2 {
            color: #FF4500;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


elif summary_section == "Limitations and Further Steps":
    st.markdown("## Limitations and Further Steps")

    st.divider()
    
    st.markdown("""
    - **Sample Size**: The data currently includes only about **a thousand records**, which may limit the variability of predictions. Larger datasets will help reduce **model overfitting** and improve **predictive accuracy**.
    - **Bike Model and Trip Duration**: The impact of **bike models** and **trip durations** hasn’t been fully explored. Analyzing **e-bike usage** versus traditional bikes can help refine **fleet management** and **bike allocation**.
    - **Station Clustering**: We can map **stations geographically** and explore clusters of **high-demand areas** to improve the **placement of new stations** or **redistribution of bikes**.
    """)
    st.markdown(
        """
        <style>
        .stMarkdown h3 {
            color: #1E90FF;
        }
        .stMarkdown h2 {
            color: #FF4500;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


