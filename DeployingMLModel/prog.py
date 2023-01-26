import numpy as np
import pickle
import streamlit as st
#filename = 'Trained_model.sav'
#pickle.dump(model, open(filename, 'wb'))

#loading the saved model
loaded_model = pickle.load(open('C:/Users/dev/OneDrive/Documents/Python Project/DeployingMLModel/Trained_model.sav', 'rb'))

def Heart_Disease_Prediction(input_data):
    # input_data = (65,1,4,120,177,0,0,140,0,0.4,1,0,7)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0] == 'Absence'):
        return 'The person dose not have heart disease'
    else:
        return 'The person have a heart disease'

def main():
    #giving a title
    st.title('Heart Disease Prediction Web App')

    #getting the input data from user
    Age = st.text_input('Enter the age')
    Sex = st.text_input('Enter the gender')
    Chest_pain_type = st.text_input('Enter the chest pain type')
    Bp = st.text_input('Enter the BP')
    Cholesterol = st.text_input('Enter the cholesterol level')
    Fbs = st.text_input('Enter the FBS over 120')
    Ekg = st.text_input('Enter the EKG result')
    Hr = st.text_input('Enter the Max HR')
    Angina = st.text_input('Enter the exercise angina')
    ST_depression = st.text_input('Enter the ST depression')
    ST_slope = st.text_input('Enter the ST slope')
    Vessels_Fluro = st.text_input('Enter the no of vessel fluro')
    Thallium = st.text_input('Enter the Thallium')

    diagnosis = ''
    if st.button('Heart Disease Result'):
        diagnosis = Heart_Disease_Prediction([[Age, Sex, Chest_pain_type, Bp, Cholesterol, Fbs, Ekg, Hr, Angina, ST_depression, ST_slope, Vessels_Fluro, Thallium]])

    st.success(diagnosis)

if __name__ == '__main__':
    main()