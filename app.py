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

st.markdown("---")

# Prediction button
if st.button("Predict Customer Segment", type="primary"):
    # Prepare input data
    input_data = np.array([[recency, frequency, monetary]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    cluster = kmeans.predict(input_scaled)[0]
    segment_name = cluster_names[cluster]
    
    # Display result
    st.success(f"### Customer Segment: *{segment_name}* (Cluster {cluster})")
    
    # Show characteristics
    st.markdown("#### Segment Characteristics:")
    
    if segment_name == "VIP Customers":
        st.info("""
        *VIP Customers* are the best customers They:
        - Shop frequently and recently
        - Spend the most money
        - Are highly engaged and loyal
        
        *Recommendation:* Reward them with exclusive offers and VIP treatment.
        """)

    elif segment_name == "Loyal Customers":
        st.info("""
        *Loyal Customers* are valuable and engaged. They:
        - Shop regularly and recently
        - Have good purchase frequency
        - Show consistent engagement
        
        *Recommendation:* Nurture with loyalty programs and early product access.
        """)

    elif segment_name == "Potential Loyalists":
        st.warning("""
        *Potential Loyalists* customers need attention! They:
        - Have moderate purchase frequency and monetary value
        
        *Recommendation:* Send win-back campaigns and special incentives.
        """)

    else:
        st.error("""
        *At Risk Customers* They:
        - Haven't purchased in a very long time.
        - Very low frequency and monetary value
        
        *Recommendation:* Send personalized win-back campaigns and special discounts to re-engage them.
        """)

# Sidebar information
st.sidebar.title("About This Project")
st.sidebar.info("""
This Customer Segmentation tool uses *K-Means Clustering* to categorize customers into distinct segments based on their 
shopping behavior (RFM Analysis).
""")