import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Predictor")

# Input features
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10)
parch = st.number_input("Parents/Children Aboard", 0, 10)
fare = st.number_input("Fare", 0.0, 500.0, 32.0)
embarked_s = st.checkbox("Embarked at Southampton?")
embarked_q = st.checkbox("Embarked at Queenstown?")

# Convert categorical features
sex_val = 1 if sex=="male" else 0
features = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_q, embarked_s]])

# Predict button
if st.button("Predict Survival"):
    pred = model.predict(features)[0]
    if pred == 1:
        st.success("✅ Survived!")
    else:
        st.error("❌ Did not survive")
