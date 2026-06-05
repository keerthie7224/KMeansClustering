import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv("Mall_Customers.csv")

# Load Models
kmeans = pickle.load(open("kmeans.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.set_page_config(
    page_title="Mall Customer Segmentation",
    layout="wide"
)

st.title("🛍 Mall Customer Segmentation")

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Dataset Overview",
        "Visualization",
        "Prediction"
    ]
)

# ------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------

if menu == "Dataset Overview":

    st.header("Dataset Overview")

    st.write(df.head())

    st.subheader("Shape")
    st.write(df.shape)

    st.subheader("Statistics")
    st.write(df.describe())

# ------------------------------------------------
# VISUALIZATION
# ------------------------------------------------

elif menu == "Visualization":

    st.header("Visualizations")

    fig, ax = plt.subplots()

    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    X = df[
        ['Annual Income (k$)',
         'Spending Score (1-100)']
    ]

    scaled = scaler.transform(X)

    clusters = kmeans.predict(scaled)

    df['Cluster'] = clusters

    fig2, ax2 = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',
        palette='Set1',
        data=df,
        ax=ax2
    )

    st.pyplot(fig2)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

else:

    st.header("Customer Cluster Prediction")

    income = st.slider(
        "Annual Income (k$)",
        10,
        150,
        50
    )

    spending = st.slider(
        "Spending Score",
        1,
        100,
        50
    )

    if st.button("Predict"):

        customer = scaler.transform(
            [[income, spending]]
        )

        cluster = kmeans.predict(customer)[0]

        st.success(
            f"Customer belongs to Cluster {cluster}"
        )