import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import pickle
from sklearn.exceptions import NotFittedError  # For handling model-related errors

# Load the model
try:
    lg = pickle.load(open('placement.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'placement.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Load and display the image
try:
    img = Image.open('Job-Placement-Agency.jpg')
    st.image(img, width=650)
except FileNotFoundError:
    st.warning("Image file 'Job-Placement-Agency.jpg' not found. Proceeding without the image.")

st.title("Job Placement Prediction Model")

# Input features
input_text = st.text_input("Enter all features (comma-separated values):")
if input_text:
    try:
        # Convert input to a NumPy array
        input_list = input_text.split(',')
        np_df = np.asarray(input_list, dtype=float)

        # Ensure the input matches the model's expected shape
        if len(np_df) != lg.n_features_in_:
            st.error(f"Expected {lg.n_features_in_} features, but got {len(np_df)}.")
        else:
            # Make prediction
            prediction = lg.predict(np_df.reshape(1, -1))

            # Display result
            if prediction[0] == 1:
                st.success("This Person Is Placed")
            else:
                st.info("This Person is not Placed")
    except ValueError:
        st.error("Invalid input. Please enter numeric values separated by commas.")
    except NotFittedError:
        st.error("The model is not properly trained. Please check the 'placement.pkl' file.")