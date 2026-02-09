import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pickle
import numpy as np

# Load model and encoders
@st.cache_data
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
st.set_option('client.showErrorDetails', False)

regressor = data['model']
le_country = data['le_country']
le_edLevel = data['le_edLevel']
le_remoteWork = data['le_remoteWork']

# UI for salary prediction
def show_predict_page():
    st.title("💼 Developers Salary Prediction")

    st.write("#### Please provide some information to help us predict your salary:")

    countries = (
        "United States of America",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "India",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Netherlands",
        "Italy",
        "Australia",
        "Poland"
    )

    education_levels = (
        "Bachelor's degree", 
        "Master's degree", 
        "Less than a Bachelors",
        "Postgraduate"
    )

    work_modes = (
        "Full in-person",
        "Fully remote",
        "Hybrid (some remote, some in-person)"
    )

    # Input widgets
    country = st.selectbox("🌍 Country", countries)
    education = st.selectbox("🎓 Education Level", education_levels)
    remoteWork = st.selectbox("🏢 Work Mode", work_modes)
    code_experience = st.slider("💻 Years of Coding Experience", 0, 50, 5)
    work_experience = st.slider("🧠 Years of Working Experience", 0, 40, 1)

    # Predict button
    if st.button("🚀 Predict Salary"):
        X = np.array([[country, education, remoteWork, code_experience, work_experience]])

        # Encode categorical values
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_edLevel.transform(X[:, 1])
        X[:, 2] = le_remoteWork.transform(X[:, 2])
        X = X.astype(float)

        # Predict and display result
        salary = regressor.predict(X)
        st.subheader(f"💰 The estimated salary is *${salary[0]:,.2f}*")