# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:53:49 2024

@author: aniru
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open("finalized_model (2).sav", 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    
    # giving a title
    st.set_page_config(page_title='Diabetes Prediction Web App', layout='wide')
    
    # getting the input data from the user
    st.sidebar.header('User Input Parameters')
    
    Pregnancies = st.sidebar.slider('Number of Pregnancies', 0, 20, 1)
    Glucose = st.sidebar.slider('Glucose Level', 0, 200, 100)
    BloodPressure = st.sidebar.slider('Blood Pressure value', 0, 122, 70)
    SkinThickness = st.sidebar.slider('Skin Thickness value', 0, 100, 20)
    Insulin = st.sidebar.slider('Insulin Level', 0, 900, 30)
    BMI = st.sidebar.slider('BMI value', 0.0, 70.0, 15.0)
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function value', 0.0, 2.5, 0.5)
    Age = st.sidebar.slider('Age of the Person', 21, 100, 21)
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        st.subheader('Prediction Result')
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
 

if __name__ == '__main__':
    main()
    




