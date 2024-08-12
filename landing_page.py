import streamlit as st

# Streamlit app for the landing page
st.set_page_config(page_title="Landing Page", layout="wide")

st.title("Welcome to the Model Training Dashboard")

st.write("""
    This dashboard allows you to train models and make predictions with different datasets. 
    Select one of the following options to get started:
""")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.write("### Navigate to the Apps")
st.sidebar.markdown("""
    - [Train and Predict with TensorFlow](pages/app.py)
    - [Train and Convert Model](pages/app1.py)
""")

# Additional content for the landing page
st.write("""
    **Train and Predict with TensorFlow**: Upload and process `homes.csv` file to train a model and make predictions.

    **Train and Convert Model**: Upload and process `result.csv` file to train a model, convert it to TensorFlow.js format, and make predictions.
""")
