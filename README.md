# 👥 Customer Segmentation Using K-Means Clustering

An end-to-end machine learning project that segments customers based on their purchasing behavior using unsupervised learning.


---

## 📖 Project Overview

This project analyzes online retail data to identify different types of customers. By grouping customers into segments, businesses can:
- Create targeted marketing campaigns
- Improve customer retention
- Personalize customer experience
- Increase sales

## 🎯 Key Features

- **RFM Analysis**: Calculate Recency, Frequency, and Monetary values
- **K-Means Clustering**: Segment customers into groups
- **Optimal Cluster Selection**: Using Elbow Method and Silhouette Score
- **PCA Visualization**: 2D visualization of customer segments
- **Interactive Dashboard**: Built with Streamlit

## 📊 Dataset

**UCI Online Retail Dataset**
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
- Size: 540,000+ transactions
- Customers: 4338
- Industry: UK-based online gift retailer
- Time Period: Dec 2010 - Dec 2011

## 🛠️ Technologies Used

- Python 3.12
- Pandas, NumPy
- Scikit-learn (K-Means, PCA, StandardScaler)
- Matplotlib, Seaborn
- Streamlit
- Jupyter Notebook



## 📂 Folder Structure

```
customer-segmentation-project/
│
├── venv/ # Virtual environment
├── data/ # Dataset
│ └── online_retail.xlsx
├── notebooks/ # Jupyter notebooks
│ └── customer_segmentation_analysis.ipynb
├── models/ # Saved models
│ ├── kmeans_model.pkl
│ ├── scaler.pkl
│ └── cluster_names.pkl
├── app/ # Streamlit web app
│ └── app.py
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```
---

## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arshadmurtaza03/customer-segmentation.git
    cd customer-segmentation
    ```

2.  **Create and activate a virtual environment:**
    ```bash
        # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

        # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Download the dataset**
- Visit [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
- Download and place `Online Retail.xlsx` in the `data/` folder

5. **Run the Jupyter Notebook**
    ```bash
    jupyter notebook notebooks/customer_segmentation.ipynb
    ```
6. **Run Streamlit App**
    ```bash
    cd app
    streamlit run app.py
    ```

---

## 📈 Results

The model identified **4 distinct customer segments**:

1. **VIP Customers**: (0.3% of total customers)
   - High value, frequent buyers, recent purchases

2. **Loyal Customers**: (4.7% of total customers)
   - Regular customers with good potential

3.  **Potential Loyalists**: (70.4% of total customers)
   - Have moderate purchase frequency and monetary value

4. **At Risk Customers**: (24.6% of customers)
   - Haven't purchased in a very long time.- Very low frequency and monetary value


**Performance:**
- PCA Variance Explained: ~85.8%

## 📚 What I Learned

- Implementing K-Means clustering from scratch
- Feature engineering with RFM analysis
- Finding optimal clusters using multiple methods
- Dimensionality reduction with PCA
- Building interactive dashboards
- Git and GitHub workflow


## How to contact

Author: Arshad Murtaza
Email: arshadmurtaza2016@gmail.com
LinkedIn: linkedin.com/in/arshadmurtaza

---