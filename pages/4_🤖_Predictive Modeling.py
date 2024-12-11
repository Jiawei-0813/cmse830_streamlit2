import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error
import time

# --- Page Setup ---
st.set_page_config(page_title="🤖 Predictive Modeling", layout="wide")

# --- Load Data ---
if "combined_data" not in st.session_state:
    st.error("Combined dataset not found in session state. Please go back and preprocess the data.")
    st.stop()

combined = st.session_state["combined_data"]

# --- Page Header ---
st.title("🤖 Predictive Modeling")
st.markdown("Analyze and predict bike-sharing patterns using various machine learning models.")

# --- Target Variable ---
target_var_column = "Number of Bike-Sharing"

if target_var_column not in combined.columns:
    # Aggregate bike counts per day if not already present
    combined[target_var_column] = 1

# --- Tabs for Model Types ---
tabs = st.tabs(["Regression Models", "Ensemble Methods", "Time Series Analysis"])

# --- Regression Models ---
with tabs[0]:
    st.header("📈 Regression Models")
    st.markdown("Perform regression analysis to predict the number of bike-sharing instances.")

    # --- Predictor and Target Variables ---
    predictors = st.multiselect(
        "Select Predictors:",
        [
            "Total duration (m)_scaled",
            "Bike Model Encoded",
            "temperature_2m_scaled",
            "apparent_temperature_scaled",
            "wind_speed_10m_scaled",
            "relative_humidity_2m_scaled",
            "Day_of_Month",
            "Day_of_Week",
            "Hour_of_Day",
        ],
        default=[
            "temperature_2m_scaled",
            "wind_speed_10m_scaled",
            "relative_humidity_2m_scaled",
            "Hour_of_Day",
        ],
    )

    if not predictors:
        st.warning("Please select at least one predictor to proceed.")
        st.stop()

    target = target_var_column

    # --- Data Preparation ---
    X = combined[predictors]
    y = combined[target]

    # Train-Test Split
    test_size = st.sidebar.slider(
        "Test Set Size", min_value=0.1, max_value=0.5, value=0.3, step=0.05, help="Adjust the proportion of the test set."
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # --- Regression Model Selection ---
    regression_type = st.sidebar.radio(
        "Select Regression Model:",
        ["Linear", "Ridge", "Lasso", "Kernel Ridge"],
        help="Choose a regression model.",
    )

    # Hyperparameter Selection
    alpha = 1.0  # Default alpha
    if regression_type in ["Ridge", "Lasso", "Kernel Ridge"]:
        alpha = st.sidebar.slider("Regularization Strength (Alpha):", 0.01, 10.0, 1.0, 0.1)

    kernel_type = "linear"  # Default kernel for Kernel Ridge
    if regression_type == "Kernel Ridge":
        kernel_type = st.sidebar.selectbox(
            "Kernel Type:", ["linear", "polynomial", "rbf"], help="Choose a kernel for Kernel Ridge Regression."
        )

    # Initialize Model
    model_map = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=alpha),
        "Lasso": Lasso(alpha=alpha),
        "Kernel Ridge": KernelRidge(alpha=alpha, kernel=kernel_type),
    }
    model = model_map[regression_type]

    # Train Model
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # --- Model Metrics ---
    metrics = {
        "R² (Train)": r2_score(y_train, y_train_pred),
        "RMSE (Train)": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "R² (Test)": r2_score(y_test, y_test_pred),
        "RMSE (Test)": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }

    st.subheader("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="R² (Train)", value=f"{metrics['R² (Train)']:.2f}")
    with col2:
        st.metric(label="RMSE (Train)", value=f"{metrics['RMSE (Train)']:.2f}")
    with col3:
        st.metric(label="R² (Test)", value=f"{metrics['R² (Test)']:.2f}")
    with col4:
        st.metric(label="RMSE (Test)", value=f"{metrics['RMSE (Test)']:.2f}")

    # --- Model Comparison ---
    with st.expander("Comparison of Regression Models"):
        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=alpha),
            "Lasso": Lasso(alpha=alpha),
            "Kernel Ridge": KernelRidge(alpha=alpha, kernel=kernel_type),
        }
        results = []
        for name, mdl in models.items():
            start_time = time.time()
            mdl.fit(X_train, y_train)
            running_time = time.time() - start_time
            y_train_pred = mdl.predict(X_train)
            y_test_pred = mdl.predict(X_test)
            results.append(
                {
                    "Model": name,
                    "R² (Train)": r2_score(y_train, y_train_pred),
                    "RMSE (Train)": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    "R² (Test)": r2_score(y_test, y_test_pred),
                    "RMSE (Test)": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                    "Running Time (s)": running_time,
                }
            )

        comparison_df = pd.DataFrame(results)
        st.dataframe(comparison_df)

        # Highlight Best Model
        best_model = comparison_df.loc[comparison_df["R² (Test)"].idxmax()]
        st.markdown("### Best Model")
        st.write(f"**Model:** {best_model['Model']}")
        st.write(f"**R² (Test):** {best_model['R² (Test)']:.2f}")
        st.write(f"**RMSE (Test):** {best_model['RMSE (Test)']:.2f}")
        st.write(f"**Running Time:** {best_model['Running Time (s)']:.2f} seconds")