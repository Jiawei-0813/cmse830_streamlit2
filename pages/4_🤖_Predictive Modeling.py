import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide",
                     page_title="🤖 Predictive Modeling"
                     )
st.title("🤖 Predictive Modeling")


st.subheader("🚲 Addressing Class Imbalance in Bike Model Prediction")

st.markdown("""
The `Bike model` variable is highly imbalanced, with the majority class (`CLASSIC`) significantly outnumbering the minority class (`PBSC_EBIKE`). 
To address this imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training dataset.
""")

# Split dataset
X = bike_2.drop(columns=["Bike model"])
y = bike_2["Bike model"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train a model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_smote, y_train_smote)

# Evaluate performance
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
st.text("Classification Report:\n")
st.text(classification_report(y_test, y_pred))