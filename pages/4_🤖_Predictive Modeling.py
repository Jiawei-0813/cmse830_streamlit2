import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Page Setup ---
st.set_page_config(page_title="🤖 Predictive Modeling", layout="wide")

# --- Load Data ---
if "combined_data" not in st.session_state:
    st.error("Combined dataset not found in session state. Please go back and preprocess the data.")
    st.stop()

combined = st.session_state["combined_data"]

# --- Page Header ---
st.title("🤖 Modeling")
st.markdown("Analyze and predict bike-sharing patterns using various machine learning models.")

# --- Target Variable ---
target_var = "Number of Bike-Sharing"

# Create the 'Number of Bike-Sharing' column by counting the number of trips per date
if target_var not in combined.columns:
    combined[target_var] = combined.groupby("Date")["Date"].transform("size")

# --- Tab Setup ---
tabs = st.tabs(["Predictive Modeling", "Time Series Analysis"])

# --- Predictive Modeling Tab ---
with tabs[0]:

    # Create a copy of the dataset to apply transformations
    combined_for_model = combined.copy()

    # --- OneHotEncoding for Time_of_Day (Top 3 Categories) ---
    # Identify top 3 categories for Time_of_Day
    top_time_of_day = combined_for_model['Time_of_Day'].value_counts().nlargest(3).index.tolist()

    # Apply the lambda function to classify 'Time_of_Day' into top 3 categories and "Other"
    combined_for_model['Time_of_Day'] = combined_for_model['Time_of_Day'].apply(lambda x: x if x in top_time_of_day else 'Other')

    # OneHotEncode 'Time_of_Day'
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    time_of_day_encoded = encoder.fit_transform(combined_for_model[['Time_of_Day']])

    # Get feature names from OneHotEncoder after fitting
    time_of_day_feature_names = encoder.get_feature_names_out(['Time_of_Day'])

    # Create DataFrame for the encoded columns
    time_of_day_df = pd.DataFrame(time_of_day_encoded, columns=time_of_day_feature_names)

    # Concatenate the new encoded columns with the original dataframe
    combined_for_model = pd.concat([combined_for_model, time_of_day_df], axis=1)

    # Show success message for encoding Time_of_Day
    st.success(f"Successfully encoded 'Time_of_Day' into binary columns for the top 3 categories: {', '.join(top_time_of_day)}.")

    # --- OneHotEncoding for Weather Description (Top 3 Categories) ---
    # Get the top 3 most frequent weather descriptions
    top_weather_codes = combined_for_model['Weather Description'].value_counts().nlargest(3).index.tolist()

    # Apply the lambda function to classify 'Weather Description' into top 3 categories and "Other"
    combined_for_model['Weather Description'] = combined_for_model['Weather Description'].apply(lambda x: x if x in top_weather_codes else 'Other')

    # OneHotEncode 'Weather Description'
    weather_desc_encoded = encoder.fit_transform(combined_for_model[['Weather Description']])

    # Get feature names from OneHotEncoder after fitting
    weather_desc_feature_names = encoder.get_feature_names_out(['Weather Description'])

    # Create DataFrame for the encoded columns
    weather_desc_df = pd.DataFrame(weather_desc_encoded, columns=weather_desc_feature_names)

    # Concatenate the new encoded columns with the original dataframe
    combined_for_model = pd.concat([combined_for_model, weather_desc_df], axis=1)

    # Show success message for encoding Weather Description
    st.success(f"Successfully encoded 'Weather Description' into binary columns for the top 3 categories: {', '.join(top_weather_codes)}.")

    # Real names for predictors
    predictor_names = {
        "Bike Model Encoded": "Bike Model",
        "temperature_2m_scaled": "Temperature (2m)",
        "apparent_temperature_scaled": "Apparent Temperature",
        "relative_humidity_2m_scaled": "Relative Humidity (2m)",
        "Day_of_Month": "Day of Month",
        "Day_of_Week": "Day of Week",
        "Hour_of_Day": "Hour of Day",
        "is_Weekend": "Is Weekend",
        "Weather Description_Overcast": "Overcast",
        "Time_of_Day_Evening": "Evening",
        "Time_of_Day_Morning": "Morning",
        "is_Rainy": "Is Rainy"
    }

    # Select predictors using real names
    predictors = st.multiselect(
        "Select Predictors:",
        list(predictor_names.values()),
        default=[
            "Temperature (2m)", 
            "Relative Humidity (2m)", 
            "Hour of Day", "Is Weekend", 
            "Overcast", 
        ],
        key="predictors"  # Adding a unique key for proper sidebar handling
    )

    # Convert selected names to actual column names for processing
    selected_predictors = [name for name in predictor_names.keys() if predictor_names[name] in predictors]

    if not selected_predictors:
        st.warning("Please select at least one predictor to proceed.")
        st.stop()

    target = target_var  # Number of bike-sharing rides

    # Data Preparation
    X = combined_for_model[selected_predictors]
    y = combined_for_model[target]

    # --- Train-Test Split ---
    test_size = st.sidebar.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.3, step=0.05, help="Adjust the proportion of the test set.", key="test_size")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # --- Model Selection and Evaluation ---
    selected_models = st.multiselect(
        "Select Models to Evaluate",
        ["Linear", "Ridge", "Lasso", "Kernel Ridge", "Random Forest", "Gradient Boosting"],
        default=["Linear", "Ridge", "Lasso", "Kernel Ridge", "Random Forest", "Gradient Boosting"],
        key="model_select"  # Adding a key for model selection in the sidebar
    )

    # Cross-validation for model comparison (K-Fold Cross Validation)
    k_folds = st.sidebar.slider("K-Fold Cross Validation", min_value=3, max_value=10, value=5, step=1, key="k_folds")

    def cross_validate(model, X, y, k_folds):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        return np.mean(cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))

    results = []

    # --- Model Training and Evaluation ---
    for model_name in selected_models:
        if model_name == "Linear":
            model = LinearRegression()
        elif model_name == "Ridge":
            with st.sidebar.expander("Ridge Parameters"):
                alpha = st.slider("Regularization (Alpha)", 0.01, 10.0, 1.0, 0.1, key="ridge_alpha")
            model = Ridge(alpha=alpha)
        elif model_name == "Lasso":
            with st.sidebar.expander("Lasso Parameters"):
                alpha = st.slider("Regularization (Alpha)", 0.01, 10.0, 1.0, 0.1, key="lasso_alpha")
            model = Lasso(alpha=alpha)
        elif model_name == "Kernel Ridge":
            with st.sidebar.expander("Kernel Ridge Parameters"):
                alpha = st.slider("Regularization (Alpha)", 0.01, 10.0, 1.0, 0.1, key="kridge_alpha")
                kernel_type = st.selectbox("Kernel Type", ["linear", "polynomial", "rbf"], key="kridge_kernel")
            model = KernelRidge(alpha=alpha, kernel=kernel_type)
        elif model_name == "Random Forest":
            with st.sidebar.expander("Random Forest Parameters"):
                n_estimators = st.slider("Number of Trees", 50, 500, 100, 50, key="rf_n_estimators")
                max_depth = st.slider("Max Depth", 1, 10, 3, key="rf_max_depth")
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_name == "Gradient Boosting":
            with st.sidebar.expander("Gradient Boosting Parameters"):
                n_estimators = st.slider("Number of Trees", 50, 500, 100, 50, key="gb_n_estimators")
                max_depth = st.slider("Max Depth", 1, 10, 3, key="gb_max_depth")
                learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key="gb_learning_rate")
            model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
        
        # Fit the model and make predictions
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Compute Metrics
        r2_train = r2_score(y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        # Cross-validation score
        cv_score = cross_validate(model, X, y, k_folds)
        
        # Adjusted R² Calculation
        adj_r2 = 1 - (1 - r2_test) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1))
        
        # Store results
        results.append({
            "Model": model_name,
            "R² (Train)": r2_train,
            "RMSE (Train)": rmse_train,
            "R² (Test)": r2_test,
            "RMSE (Test)": rmse_test,
            "MAE (Test)": mae_test,
            "MSE (Test)": mse_test,
            "Adjusted R² (Test)": adj_r2,
            "Cross-Validation MSE": -cv_score,
        })

    # --- Display Results ---
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # --- Highlight Best Model ---
    with st.expander("Best Model Metrics", expanded=True):
        best_model = results_df.loc[results_df["R² (Test)"].idxmax()]
        st.metric("Model", best_model['Model'])

        col1, col2, col3 = st.columns(3)
        
        col1.metric("R² (Test)", f"{best_model['R² (Test)']:.2f}")
        col1.metric("Adjusted R² (Test)", f"{best_model['Adjusted R² (Test)']:.2f}")
        
        col2.metric("RMSE (Test)", f"{best_model['RMSE (Test)']:.2f}")
        col2.metric("MAE (Test)", f"{best_model['MAE (Test)']:.2f}")
        
        col3.metric("Cross-Validation MSE", f"{best_model['Cross-Validation MSE']:.2f}")

        # Auto-generated explanation about the best model
        if best_model['R² (Test)'] > 0.8:
            st.write("This model explains a significant amount of the variance in bike-sharing trips, with an R² greater than 0.8.")
        elif best_model['R² (Test)'] > 0.5:
            st.write("The model has a reasonable performance with moderate accuracy in explaining bike-sharing patterns.")
        else:
            st.write("The model's performance could be improved, as it explains less than 50% of the variance in the target variable.")

        st.divider()
        
        # --- Plot Actual vs Predicted with Hover ---
        import plotly.graph_objects as go

        st.subheader("Plot: Actual vs Predicted")
        fig = go.Figure()

        # Add actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_test[:100]))),
            y=y_test[:100],
            mode='lines+markers',
            name='Actual Values',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        # Add predicted values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_test_pred[:100]))),
            y=y_test_pred[:100],
            mode='lines+markers',
            name='Predicted Values',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

        fig.update_layout(
            xaxis_title="Data Points (Test Set)",
            yaxis_title="Number of Bike-Sharing Trips",
            legend_title="Legend"
        )

        st.plotly_chart(fig)
        
        st.divider()

        # Displaying coefficients or feature importance if available
        if hasattr(model, 'coef_'):
            # For linear models, display coefficients
            st.subheader("Model Coefficients")
            coef_df = pd.DataFrame({
                "Predictor": selected_predictors,
                "Coefficient": model.coef_
            })
            st.dataframe(coef_df)
            st.write("Model coefficients show the influence of each predictor on the target variable. Larger absolute values indicate stronger influence.")
            
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models, display feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                "Feature": selected_predictors,
                "Importance": model.feature_importances_
            })

            col1, col2 = st.columns([1,2])
            
            with col1:
                st.dataframe(feature_importance)
                 # Plot feature importance for tree-based models (Random Forest, Gradient Boosting)
            if model_name in ["Random Forest", "Gradient Boosting"]:
                with col2:
                    fig, ax = plt.subplots(figsize=(14, 6))
                    feature_importance.plot(kind='barh', x='Feature', y='Importance', ax=ax, color='lightblue')
                    ax.set_title(f"Feature Importance for {model_name}", fontsize=20)
                    ax.set_xlabel(None, fontsize=14)
                    ax.set_ylabel(None, fontsize=14)
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    st.pyplot(fig)
            st.write("Feature importance values indicate each feature's contribution to the model's predictions. Higher values suggest greater influence.")
           
# --- Time Series Analysis Tab ---
with tabs[1]:
    # --- Ensure 'Date' column is in datetime format and set it as index ---
    combined_for_model['Date'] = pd.to_datetime(combined_for_model['Date'])
    combined_for_model.set_index('Date', inplace=True)

    # --- Resample data by day and sum ---
    daily_data = combined_for_model.resample('D').sum()

    # Display the daily data or plot as needed
    if 'Number of Bike-Sharing' in daily_data.columns:
        st.line_chart(daily_data['Number of Bike-Sharing'])
    else:
        st.error("'Number of Bike-Sharing' column is missing or misformatted.")

    st.divider()

    # --- Time Series Decomposition ---
    st.subheader("Time Series Decomposition")
    if len(daily_data) >= 60:  # Ensure there are enough data points for seasonal decomposition
        decomposition = seasonal_decompose(daily_data['Number of Bike-Sharing'], model='additive', period=30)
        fig, ax = plt.subplots(figsize=(8, 4))
        decomposition.plot(ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Insufficient data for seasonal decomposition. The dataset must contain at least 60 observations.")

    st.divider()

    # --- ARIMA Model Setup ---
    st.subheader("ARIMA Model Forecast")
    p = st.slider('AR term (p)', 0, 5, 1)
    d = st.slider('Differencing term (d)', 0, 2, 1)
    q = st.slider('MA term (q)', 0, 5, 1)

    arima_model = ARIMA(daily_data['Number of Bike-Sharing'], order=(p, d, q))
    arima_fit = arima_model.fit()

    # Display ARIMA model summary
    with st.expander("ARIMA Model Summary"):
        st.text(str(arima_fit.summary()))

        # Forecasting
        n_periods = st.slider('Forecast Periods', 1, 30, 7)
        arima_forecast = arima_fit.forecast(steps=n_periods, alpha=0.05)  # Only forecast, no stderr or conf_int

        arima_forecast_index = pd.date_range(start=daily_data.index[-1], periods=n_periods + 1, freq='D')[1:]
        arima_forecast_df = pd.DataFrame({'Forecast': arima_forecast}, index=arima_forecast_index)

        # Plot ARIMA forecast
        st.subheader("ARIMA Forecast Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Number of Bike-Sharing'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=arima_forecast_df.index, y=arima_forecast_df['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='red')))
        fig.update_layout(title='ARIMA Forecast', xaxis_title='Date', yaxis_title='Number of Bike-Sharing Rides')
        st.plotly_chart(fig)

        st.divider()

    # --- Exponential Smoothing (ETS) Setup ---
    st.subheader("Exponential Smoothing Model Forecast")

    ets_model = ExponentialSmoothing(daily_data['Number of Bike-Sharing'], trend='add', seasonal='add', seasonal_periods=7)
    ets_fit = ets_model.fit()

    # Display ETS model summary
    with st.expander("ETS Model Summary"):
        st.text(str(ets_fit.summary()))

        # Forecasting
        ets_forecast = ets_fit.forecast(steps=n_periods)
        ets_forecast_index = pd.date_range(start=daily_data.index[-1], periods=n_periods + 1, freq='D')[1:]
        ets_forecast_df = pd.DataFrame({'Forecast': ets_forecast}, index=ets_forecast_index)

        # Plot ETS forecast
        st.subheader("ETS Forecast Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Number of Bike-Sharing'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=ets_forecast_df.index, y=ets_forecast_df['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='green')))
        fig.update_layout(title='Exponential Smoothing (ETS) Forecast', xaxis_title='Date', yaxis_title='Number of Bike-Sharing Rides')
        st.plotly_chart(fig)

    st.divider()

    # --- Model Evaluation ---
    st.subheader("Forecast Evaluation")
    if n_periods <= len(daily_data):
        # Calculate RMSE, MAE, and MSE for both models
        actual = daily_data.iloc[-n_periods:]['Number of Bike-Sharing']
        
        arima_rmse = np.sqrt(mean_squared_error(actual, arima_forecast[:n_periods]))
        ets_rmse = np.sqrt(mean_squared_error(actual, ets_forecast[:n_periods]))
        
        arima_mae = mean_absolute_error(actual, arima_forecast[:n_periods])
        ets_mae = mean_absolute_error(actual, ets_forecast[:n_periods])

        arima_mse = mean_squared_error(actual, arima_forecast[:n_periods])
        ets_mse = mean_squared_error(actual, ets_forecast[:n_periods])

        # Display RMSE, MAE, MSE for both models
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ARIMA Model")
            st.write(f"RMSE: {arima_rmse:.2f}")
            st.write(f"MAE: {arima_mae:.2f}")
            st.write(f"MSE: {arima_mse:.2f}")
        
        with col2:
            st.subheader("ETS Model")
            st.write(f"RMSE: {ets_rmse:.2f}")
            st.write(f"MAE: {ets_mae:.2f}")
            st.write(f"MSE: {ets_mse:.2f}")

        # Auto-generated comparison text
        if arima_rmse < ets_rmse:
            st.write("**ARIMA provides a better forecast with lower RMSE, MAE, and MSE compared to ETS.**")
        elif ets_rmse < arima_rmse:
            st.write("**ETS provides a better forecast with lower RMSE, MAE, and MSE compared to ARIMA.**")
        else:
            st.write("**Both models provide similar performance. Consider the model's characteristics and the specific use case.**")

    # Optionally save the models to session state if further analysis is needed
    st.session_state['arima_model'] = arima_fit
    st.session_state['ets_model'] = ets_fit