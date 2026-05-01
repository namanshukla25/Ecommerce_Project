import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("📊 Ecommerce Data Analysis Dashboard")

# Load data
df = pd.read_csv("output/cleaned_data.csv")

# Sidebar filters
st.sidebar.header("Filters")

region = st.sidebar.selectbox("Select Region", df["region"].unique())
category = st.sidebar.selectbox("Select Category", df["category"].unique())

# Filter data
filtered_df = df[(df["region"] == region) & (df["category"] == category)]

# KPIs
st.subheader("📌 Key Metrics")
st.write("Total Sales:", filtered_df["sales"].sum())
st.write("Total Profit:", filtered_df["profit"].sum())

# Sales by Sub-Category
st.subheader("📊 Sales by Sub-Category")
fig, ax = plt.subplots()
filtered_df.groupby("sub_category")["sales"].sum().plot(kind="bar", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Profit vs Sales
st.subheader("📈 Profit vs Sales")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=filtered_df, x="sales", y="profit", ax=ax2)
st.pyplot(fig2)