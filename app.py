import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

# Load the trained models
linear_model = joblib.load('linear_model.pkl')
lasso_model = joblib.load('lasso_model.pkl')
bagging_model = joblib.load('bagging_lr.pkl')
stacking_model = joblib.load('stacking_model.pkl')
model_Neural_Net = keras.models.load_model('model_Neural_Net.h5')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Mapping from numerical predictions to weather categories
weather_mapping = {0: 'drizzle', 1: 'fog', 2: 'rain', 3: 'snow', 4: 'sun'}

# Streamlit UI
st.title("Weather Prediction App")

st.write("""
### Enter weather data to predict:
- Precipitation
- Max Temperature
- Min Temperature
- Wind Speed
""")

# Input fields
precipitation = st.number_input("Precipitation", min_value=0.0, value=0.0, format="%.2f")
temp_max = st.number_input("Max Temperature", min_value=-30.0, value=10.0, format="%.2f")
temp_min = st.number_input("Min Temperature", min_value=-30.0, value=0.0, format="%.2f")
wind = st.number_input("Wind Speed", min_value=0.0, value=5.0, format="%.2f")

# Prepare input data
input_data = np.array([[precipitation, temp_max, temp_min, wind]])
scaled_input = scaler.transform(input_data)

# Perform predictions when button is clicked
if st.button('Predict Weather'):
    st.write("### Predictions:")
    
    # Neural Network Prediction
    nn_pred = model_Neural_Net.predict(scaled_input)
    nn_class = np.argmax(nn_pred)
    nn_output = weather_mapping.get(nn_class, 'Unknown')
    st.write(f"- Neural Network: **{nn_output}**")
    
    # Linear Regression Prediction
    linear_pred = linear_model.predict(scaled_input)
    linear_class = int(round(linear_pred[0]))
    linear_output = weather_mapping.get(linear_class, 'Unknown')
    st.write(f"- Linear Regression: **{linear_output}**")
    
    # Lasso Regression Prediction
    lasso_pred = lasso_model.predict(scaled_input)
    lasso_class = int(round(lasso_pred[0]))
    lasso_output = weather_mapping.get(lasso_class, 'Unknown')
    st.write(f"- Lasso Regression: **{lasso_output}**")
    
    # Bagging Model Prediction
    bagging_pred = bagging_model.predict(scaled_input)
    bagging_class = int(round(bagging_pred[0]))
    bagging_output = weather_mapping.get(bagging_class, 'Unknown')
    st.write(f"- Bagging Model: **{bagging_output}**")
    
    # Stacking Model Prediction
    stacking_pred = stacking_model.predict(scaled_input)
    stacking_class = int(round(stacking_pred[0]))
    stacking_output = weather_mapping.get(stacking_class, 'Unknown')
    st.write(f"- Stacking Model: **{stacking_output}**")
