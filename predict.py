import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_gender = data['le_gender']
le_marriage = data['le_marriage']
le_work = data['le_work']
le_smoking = data['le_smoking']

def show_predict_page():
    st.title("Cerebral Stroke Prediction")

    st.write("""### We need some information to predict your Cerebral Stroke Chances""")

    gender = {
        'Male', 
        'Female', 
        'Other'
    }

    marriage = {
        'Yes', 
        'No'
    }

    work = {
        'Private', 
        'Self-employed', 
        'Govt_job', 
        'children', 
        'Never_worked'
    }

    smoking = {
        'never smoked', 
        'formerly smoked', 
        'smokes'
    }

    gender = st.selectbox("Gender", gender)
    age = st.slider("Select your age", 30, 80, 35)
    marriage = st.selectbox("Marriage", marriage)
    work = st.selectbox("Work", work)
    glucoselvl = st.number_input(
        "Please enter your Average Glucose Level", min_value=55.0, max_value=290.0, value=106.0, step=.1 
    )
    bmi = st.number_input(
        "Please enter your BMI", min_value=10.0, max_value=92.0, value=25.0, step=.1
    )
    smoking = st.selectbox("Smoking", smoking)
    hypertension = st.checkbox("Please select the checkbox if you have Hypertension")
    if hypertension == False:
        ht = 0.0
    else:
        ht = 1.0
    heart_disease = st.checkbox("Please select the checkbox if you have Heart Disease")
    if heart_disease == False:
        hd = 0.0
    else:
        hd = 1.0
    ok = st.button("Predict Cerebral Stroke Tendency")
    if ok:
        X = np.array([[gender, age, ht, hd, marriage, work, glucoselvl, bmi, smoking]])
        X[:, 0] = le_gender.transform(X[:, 0])
        X[:, 4] = le_marriage.transform(X[:, 4])
        X[:, 5] = le_work.transform(X[:, 5])
        X[:, 8] = le_smoking.transform(X[:, 8])
        X = X.astype(float)

        verdict = regressor_loaded.predict(X)
        
        if verdict[0] == 0:
            st.subheader("You do not have any Cerebral Stroke Tendency")
        else:
            st.subheader("You have high Cerebral Stroke Tendency")
        
        