import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st
from tensorflow.keras.models import load_model


model = load_model('model.h5')


with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

st.title('Customer Churn Prediction')
geography = st.selectbox('Geography', onehot_encoder.categories_[0].tolist())
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary')

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : label_encoder.transform([gender])[0],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_credit_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary],
})

geography_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out())
input_data = pd.concat([input_data, geo_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write(f'Customer is likely to churn with a probability of {prediction_probability:.2f}')
else:
    st.write(f'Customer is not likely to churn with a probability of {1 - prediction_probability:.2f}')