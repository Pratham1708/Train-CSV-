import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# Function to convert the model to TensorFlow.js format
def convert_model():
    os.system('tensorflowjs_converter --input_format=tf.keras --output_format=tfjs_graph_model my_model.h5 model')

# Streamlit app
st.title("Train and Convert TensorFlow Model")

# File upload
uploaded_file = r"result.csv"
if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(df.head())  # Show the first few rows of the dataset

    st.write("### Basic Statistics:")
    st.write(df.describe())  # Show basic statistics of the dataset

    # Input fields for column names
    feature_columns = st.text_input('Enter the column names for features (X) (comma-separated):', 'Age, Salary, Experience')
    target_column = st.text_input('Enter the column name for target (y):', 'Purchase')

    feature_columns_list = [col.strip() for col in feature_columns.split(',')]

    # Ensure the dataset contains the specified columns
    if all(col in df.columns for col in feature_columns_list) and target_column in df.columns:
        X = df[feature_columns_list].values  # Extract feature data
        y = df[target_column].values  # Extract target data

        # Train model button
        if st.button('Train Model'):
            st.write("Training model...")
            model = train_model(X, y)  # Train the model
            st.write("Model trained and saved as 'my_model.h5'.")

            st.write("Converting model to TensorFlow.js format...")
            convert_model()  # Convert the model to TensorFlow.js format

            st.write("Model converted and saved in 'model' directory.")
            st.success("Process complete!")

            # Set a flag indicating the model is trained
            st.session_state.model_trained = True
    else:
        st.write("<p class='warning'>Dataset must contain the specified columns for training.</p>", unsafe_allow_html=True)

# Initialize session state for model training
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Prediction and visualization
if st.session_state.model_trained:
    st.write("### Make Predictions and Show Graphs:")

    if st.button('Predict and Show Graphs'):
        st.title("Automatic Predictions and Visualization")
        # Load the trained model
        model = tf.keras.models.load_model('my_model.h5')

        # Prepare input data (all features)
        X = df[feature_columns_list].values

        # Make predictions for the entire dataset
        predictions = model.predict(X)

        # Add predictions to the dataframe
        df['Predicted_Purchase'] = predictions

        # Scatter Plot: Age vs. Salary with Predicted Purchase outcome
        st.write("### Scatter Plot: Age vs. Salary with Predicted Purchase outcome")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x='Age', y='Salary', hue='Predicted_Purchase', data=df, ax=ax1, palette='coolwarm')
        st.pyplot(fig1)

        # Histogram: Distribution of Predicted Purchase outcomes
        st.write("### Histogram: Distribution of Predicted Purchase outcomes")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Predicted_Purchase'], bins=10, kde=True, ax=ax2, color='skyblue')
        st.pyplot(fig2)

        # Correlation Heatmap: Including Predicted_Purchase
        st.write("### Correlation Heatmap with Predicted Purchase")
        fig3, ax3 = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax3)
        st.pyplot(fig3)

        st.write("### Line Plot: Predictions vs Index")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(df.index, df['Predicted_Purchase'], label='Predicted Purchase', marker='o', linestyle='-', color='orange')
        ax4.set_title('Line Plot: Predictions vs Index')
        ax4.set_xlabel('Index')
        ax4.set_ylabel('Predicted Purchase')
        ax4.legend()
        st.pyplot(fig4)
        st.success("Predictions made and graphs generated!")

else:
    st.write("Please upload a CSV file and train the model first.")
