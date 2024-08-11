import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to train a simple model
def train_model(X, y):
    # Define a simple neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=10, activation='relu', input_shape=[X.shape[1]]),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=0)  # Train the model
    model.save('my_model.h5')  # Save the trained model with .h5 extension
    return model

# Streamlit app
st.set_page_config(page_title="Train and Predict with TensorFlow", layout="wide")
st.title("Train and Predict with TensorFlow Model")

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .title {
#         color: #333;
#         text-align: center;
#     }
#     .sidebar .sidebar-content {
#         background-color: #e1e1e1;
#     }
#     .warning {
#         color: red;
#     }
#     .success {
#         color: green;
#     }
#     </style>
# """, unsafe_allow_html=True)

# File upload
uploaded_file = r"homes.csv"

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(df.head())  # Show the first few rows of the dataset

    st.write("### Basic Statistics:")
    st.write(df.describe())  # Show basic statistics of the dataset
    
    # Display available columns for debugging
    st.write("### Available Columns in CSV:")
    st.write(df.columns.tolist())  # Print available columns

    # Input fields for column names
    feature_columns = st.text_input('Enter the column names for features (X) (comma-separated):', 'Living, Rooms, Beds, Baths, Age, Acres, Taxes')
    target_column = st.text_input('Enter the column name for target (y):', 'Sell')

    feature_columns_list = [col.strip() for col in feature_columns.split(',')]

    # Ensure the dataset contains the specified columns
    if all(col in df.columns for col in feature_columns_list) and target_column in df.columns:
        X = df[feature_columns_list].values  # Extract feature data
        y = df[target_column].values  # Extract target data

        if st.button('Train Model'):
            st.write("Training model...")
            model = train_model(X, y)  # Train the model
            st.write("Model trained and saved as 'my_model.h5'.")

            st.write("Model converted and saved in 'model' directory.")
            st.success("Process complete!")

            # Set a flag indicating the model is trained
            st.session_state.model_trained = True
    else:
        st.write("<p class='warning'>Dataset must contain the specified columns for training.</p>", unsafe_allow_html=True)

    # Prediction
    if st.button('Predict and Show Graphs'):
        if os.path.exists('my_model.h5'):
            # Load the trained model
            model = tf.keras.models.load_model('my_model.h5')

            # Prepare input data (all features)
            X = df[feature_columns_list].values

            # Make predictions for the entire dataset
            predictions = model.predict(X)
            df['Predicted_Sell'] = predictions

            # Scatter Plot: Age vs. Taxes with Predicted Sell outcome
            st.write("### Scatter Plot: Age vs. Taxes with Predicted Sell outcome")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            sns.scatterplot(x='Age', y='Taxes', hue='Predicted_Sell', data=df, ax=ax1, palette='coolwarm')
            ax1.set_title('Scatter Plot: Age vs. Taxes')
            st.pyplot(fig1)

            # Histogram: Distribution of Predicted Sell outcomes
            st.write("### Histogram: Distribution of Predicted Sell outcomes")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            sns.histplot(df['Predicted_Sell'], bins=10, kde=True, ax=ax2, color='skyblue')
            ax2.set_title('Histogram: Distribution of Predicted Sell Outcomes')
            st.pyplot(fig2)

            # Correlation Heatmap: Including Predicted_Sell
            st.write("### Correlation Heatmap with Predicted Sell")
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax3)
            ax3.set_title('Correlation Heatmap with Predicted Sell')
            st.pyplot(fig3)

            # Line Plot: Predictions vs Actual Values
            st.write("### Line Plot: Predictions vs Actual Values")
            fig4, ax4 = plt.subplots(figsize=(6, 6))
            ax4.plot(df.index, df[target_column], label='Actual', marker='o')
            ax4.plot(df.index, df['Predicted_Sell'], label='Predicted', marker='x')
            ax4.set_title('Line Plot: Predictions vs Actual Values')
            ax4.set_xlabel('Index')
            ax4.set_ylabel('Sell')
            ax4.legend()
            st.pyplot(fig4)

            st.markdown("<p class='success'>Predictions made and graphs generated!</p>", unsafe_allow_html=True)

        else:
            st.markdown("<p class='warning'>Please train the model first.</p>", unsafe_allow_html=True)

else:
    st.error("Please upload a CSV file.")
