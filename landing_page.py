import streamlit as st

# Streamlit app for the landing page
st.set_page_config(page_title="Landing Page", layout="wide")

st.title("Welcome to the Model Training Dashboard")

st.write("""
    This dashboard allows you to train models and make predictions with different datasets. 
    Select one of the following options to get started:
""")

# Navigation links
st.write("### Navigate to the Apps")
st.markdown("""
    - [Train and Predict Model with homes.csv](http://localhost:8501/?app=app.py)
    - [Train and Convert Model with result.csv](http://localhost:8502/?app=app1.py)
""")

st.write("""
    **Train and Predict Model with homes.csv**: Upload and process `homes.csv` file to train a model and make predictions.

    **Train and Convert Model with result.csv**: Upload and process `result.csv` file to train a model, convert it to TensorFlow.js format, and make predictions.
""")
