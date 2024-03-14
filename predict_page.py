import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
le_country = data['le_country']
le_education = data['le_education']

def show_predict_page():
    st.image("header.png")

    st.write('''### 💼 Enter your Details below to Predict your Salary 💰''')
    countries = (
    "United States of America",
    "Other",
    "Germany",
    "United Kingdom of Great Britain and Northern Ireland",
    "Canada",
    "India",
    "France",
    "Netherlands",
    "Australia",
    "Brazil",
    "Spain",
    "Sweden",
    "Italy",
    "Poland",
    "Switzerland",
    "Denmark",
    "Norway",
    "Israel")

    education = ("Bachelor's degree", 'Less than a Bachelors', "Master's degree",
       'Post grad')

    country = st.selectbox('Country',countries)
    education = st.selectbox('Education',education)
    experience = st.slider('Years of Experience',0,  50 , 3) #start, stop, default start

    ok = st.button('Calculate Salary')
    if ok:
        X = np.array([[country,education, experience]])
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        X[:, 0] = le.fit_transform(X[:, 0])
        X[:, 1] = le.fit_transform(X[:, 1])
        X = X.astype(float)
        salary = regressor.predict(X)
        st.subheader(f'The estimated salary is ${salary[0]:.2f}')
