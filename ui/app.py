
import streamlit as st
import pandas as pd
import joblib
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "final_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)

except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("Heart Disease Risk Prediction")

st.write(
    """
    This application uses a machine learning model to estimate
    the likelihood of heart disease based on patient information.

    Enter the patient's clinical information below and click
    Predict to generate a prediction.
    """
)

st.warning(
    "This application is for educational and demonstration purposes "
    "only and should not be used as a medical diagnosis."
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.header("Patient Information")

col1, col2, col3 = st.columns(3)


# =========================================================
# COLUMN 1
# =========================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
        "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=50,
        max_value=700,
        value=240
    )


# =========================================================
# COLUMN 2
# =========================================================

with col2:

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )


# =========================================================
# COLUMN 3
# =========================================================

with col3:

    slope = st.selectbox(
        "Slope of Peak Exercise ST",
        options=[1, 2, 3]
    )

    ca = st.selectbox(
        "Number of Major Vessels (CA)",
        options=[0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thalassemia",
        options=[3, 6, 7]
    )


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})


# =========================================================
# VIEW INPUT DATA
# =========================================================

with st.expander("View Input Data"):

    st.dataframe(
        input_data,
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "Predict Heart Disease Risk",
    use_container_width=True
):

    try:

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_data)[0]

            disease_probability = probability[1] * 100
            no_disease_probability = probability[0] * 100

        else:

            disease_probability = None
            no_disease_probability = None


        st.divider()

        st.header("Prediction Result")


        # =================================================
        # RESULT
        # =================================================

        if prediction == 1:

            st.error(
                "Higher likelihood of heart disease"
            )

        else:

            st.success(
                "Lower likelihood of heart disease"
            )


        # =================================================
        # PROBABILITY
        # =================================================

        if disease_probability is not None:

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "No Disease Probability",
                    f"{no_disease_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Disease Probability",
                    f"{disease_probability:.2f}%"
                )

            st.subheader(
                "Heart Disease Probability"
            )

            st.progress(
                min(int(disease_probability), 100)
            )


        # =================================================
        # DISCLAIMER
        # =================================================

        st.info(
            """
            The prediction is generated by the trained machine
            learning model and is intended for educational and
            demonstration purposes only. It is not a medical diagnosis.
            """
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# =========================================================
# DATA EXPLORATION
# =========================================================

st.divider()

st.header("Heart Disease Data Exploration")

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "heart_disease.csv"
)

try:

    df = pd.read_csv(DATA_PATH)

    tab1, tab2, tab3 = st.tabs([
        "Age Distribution",
        "Disease Distribution",
        "Cholesterol vs Age"
    ])


    # =====================================================
    # AGE DISTRIBUTION
    # =====================================================

    with tab1:

        st.subheader("Age Distribution")

        age_counts = (
            df["age"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(age_counts)


    # =====================================================
    # DISEASE DISTRIBUTION
    # =====================================================

    with tab2:

        st.subheader("Heart Disease Distribution")

        disease_counts = (
            df["target"]
            .value_counts()
            .sort_index()
        )

        disease_counts.index = [
            "No Disease" if x == 0 else "Disease"
            for x in disease_counts.index
        ]

        st.bar_chart(disease_counts)


    # =====================================================
    # CHOLESTEROL VS AGE
    # =====================================================

    with tab3:

        st.subheader("Cholesterol vs Age")

        cholesterol_age = df[
            ["age", "chol"]
        ].sort_values("age")

        st.line_chart(
            cholesterol_age.set_index("age")
        )


except Exception as e:

    st.error(
        f"Could not load dataset: {e}"
    )
