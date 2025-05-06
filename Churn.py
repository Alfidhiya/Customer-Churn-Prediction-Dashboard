import pandas as pd
import numpy as np

# Import Model Final
from xgboost import XGBClassifier

# Load Model
import pickle
import joblib

import streamlit as st

# ====================================================================================================

# Judul Utama
st.write("""
         <div style="text-align: center;">
         <h1> 💸 Customer Churn Prediction 📉 </h1>
         """, unsafe_allow_html=True)

# Side Bar Menu
st.sidebar.header("Please Input Your Customer Features")

def user_input_features():
    CreditScore = st.sidebar.slider(label= 'CreditScore', 
                    min_value = 350,
                    max_value = 850, value = 500)
    Balance = st.sidebar.slider(label = 'Balance',
                            min_value = 0,
                            max_value = 260000, value = 130000)
    EstimatedSalary = st.sidebar.slider(label = 'EstimatedSalary',
                            min_value = 11,
                            max_value = 200000, value = 100000)
    Age = st.sidebar.number_input(label = 'Age',
                            min_value = 18,
                            max_value = 92, value = 30)
    Tenure = st.sidebar.number_input(label = 'Tenure',
                            min_value = 0,
                            max_value = 10, value = 5)
    NumOfProducts = st.sidebar.number_input(label = 'NumOfProducts',
                            min_value = 1,
                            max_value = 5, value = 2)
    HasCrCard = st.sidebar.selectbox(label = 'HasCrCard',
                        options = [0, 1])
    IsActiveMember = st.sidebar.selectbox(label = 'IsActiveMember',
                        options = [0, 1])
    Gender = st.sidebar.selectbox(label = 'Gender',
                        options = ['Female', 'Male'])
    Geography = st.sidebar.selectbox(label = 'Geography',
                        options = ['France', 'Germany', 'Spain'])
    df = pd.DataFrame()
    df['CreditScore'] = [CreditScore]
    df['Geography'] = [Geography]
    df['Gender'] = [Gender]
    df['Age'] = [Age]
    df['Tenure'] = [Tenure]
    df['Balance'] = [Balance]
    df['NumOfProducts'] = [NumOfProducts]
    df['HasCrCard'] = [HasCrCard]
    df['IsActiveMember'] = [IsActiveMember]
    df['EstimatedSalary'] = [EstimatedSalary]
    return df

df_features = user_input_features()

# Memanggil Model
model = pickle.load(open('model_xgboost.sav', 'rb'))

# Predict
prediction = model.predict(df_features)

# Deskripsi Dashboard
st.markdown("""
<div style='text-align: center'>
Dashboard ini dirancang untuk membantu pihak manajemen bank dalam memprediksi kemungkinan seorang nasabah akan melakukan <b>churn</b> 
(berhenti menggunakan layanan bank), berdasarkan karakteristik dan perilaku mereka. <br><br>
Dengan informasi ini, bank dapat mengambil langkah-langkah preventif untuk mempertahankan nasabah yang memiliki risiko churn yang tinggi.
</div> <br><br>
            
""", unsafe_allow_html=True)

# Membuat Kolom
col1, col2 = st.columns(2)

with col1:
    # Mengeluarkan Data Frame
    st.subheader('Customer Characteristics') 
    st.write(df_features.transpose())

with col2:
    # Mengeluarkan Hasil Prediksi
    st.subheader('Prediction Result')
    if prediction == 1:
        st.write("""
        <div style="text-align: center;">
        <h2 style="color: red;">😩 CHURN 😩</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write("""
        <div style="text-align: center;">
        <h2 style="color: green;">💚 STAY 💚</h2>
        </div>
        """, unsafe_allow_html=True)