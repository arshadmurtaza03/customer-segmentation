import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="centered"
)


# Load the saved models
@st.cache_resource
def load_models():
    with open('models/kmeans_model.pkl', 'rb') as file:
        kmeans = pickle.load(file)
    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('models/cluster_names.pkl', 'rb') as file:
        cluster_names = pickle.load(file)
    
    return kmeans, scaler, cluster_names


kmeans, scaler, cluster_names = load_models()


# Title and description
st.title("👥 Customer Segmentation Predictor")
st.markdown("""
This tool predicts which customer segment a person belongs to based on their purchasing behavior.
Enter the customer details below and click **Predict Segment** to see the results.
""")


st.markdown("---")


# Create input form
st.subheader("Enter Customer Information")


col1, col2 = st.columns(2)


with col1:
    recency = st.number_input(
        "Recency (Days since last purchase)",
        min_value=0,
        max_value=500,
        value=30,
        help="How many days ago did this customer last make a purchase?"
    )
    
    frequency = st.number_input(
        "Frequency (Number of purchases)",
        min_value=1,
        max_value=100,
        value=5,
        help="How many times has this customer made a purchase?"
    )


with col2:
    monetary = st.number_input(
        "Monetary (Total amount spent)",
        min_value=0.0,
        max_value=500000.0,
        value=500.0,
        step=10.0,
        help="What is the total amount this customer has spent?"
    )
