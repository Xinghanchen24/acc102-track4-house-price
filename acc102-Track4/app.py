# ==========================================================
# ACC102 Track4: Interactive House Price Analysis Tool
# Student Name: [Your Name]
# Student ID: [Your ID]
# Dataset: Ames Housing Dataset (Kaggle)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Ames Housing Price Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Style & Header ----------------------
st.title(" Ames Housing Price Analysis & Prediction Tool")
st.markdown("### An Interactive Data Product for ACC102")
st.divider()

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df = df.drop(columns=["Id"], errors="ignore")
    return df

df = load_data()

# ---------------------- Sidebar Navigation ----------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Dataset Overview",
        "Missing Value Analysis",
        "Outlier Detection",
        "Data Visualization",
        "Price Prediction Model"
    ]
)

# ---------------------- 1. Dataset Overview ----------------------
if section == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Descriptive Statistics"):
        st.dataframe(df.describe(), use_container_width=True)

# ---------------------- 2. Missing Value Analysis ----------------------
elif section == "Missing Value Analysis":
    st.subheader(" Missing Value Analysis")

    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing Percentage": missing_pct.round(2)
    })

    st.dataframe(missing_df, use_container_width=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    missing_pct.plot(kind="bar", color="#ff4b4b", ax=ax)
    ax.set_title("Missing Value Percentage by Feature")
    ax.set_ylabel("Percentage (%)")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------- 3. Outlier Detection ----------------------
elif section == "Outlier Detection":
    st.subheader("Outlier Detection (GrLivArea vs SalePrice)")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.6, color="#3d85c6")
    ax.set_xlabel("Above Ground Living Area (sq ft)")
    ax.set_ylabel("Sale Price ($)")
    ax.set_title("Before Outlier Removal")
    st.pyplot(fig)

    # Remove outliers
    df_clean = df[df["GrLivArea"] < 4000]
    st.success(f"Before: {len(df)} | After cleaning: {len(df_clean)}")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.scatter(df_clean["GrLivArea"], df_clean["SalePrice"],
                alpha=0.6, color="#2ecc71")
    ax2.set_xlabel("Above Ground Living Area (sq ft)")
    ax2.set_ylabel("Sale Price ($)")
    ax2.set_title("After Outlier Removal")
    plt.tight_layout()
    st.pyplot(fig2)

# ---------------------- 4. Professional Visualization ----------------------
elif section == "Data Visualization":
    st.subheader(" Data Visualization & Insights")

    # 1. Sale Price Distribution
    st.markdown("#### 1. Sale Price Distribution")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.histplot(df["SalePrice"], kde=True, color="#f39c12", ax=ax)
    ax.set_title("Distribution of House Prices")
    plt.tight_layout()
    st.pyplot(fig)

    # 2. Correlation Heatmap
    st.markdown("#### 2. Feature Correlation Heatmap")
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    # 3. Year Built vs Price
    st.markdown("#### 3. Year Built vs Sale Price")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.lineplot(x=df["YearBuilt"], y=df["SalePrice"], color="#9b59b6", ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    # 4. Overall Quality vs Price
    st.markdown("#### 4. Overall Quality vs Sale Price")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.boxplot(x=df["OverallQual"], y=df["SalePrice"], palette="viridis", ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------- 5. Prediction Model ----------------------
elif section == "Price Prediction Model":
    st.subheader("Linear Regression Price Prediction Model")

    # Data cleaning
    df_model = df[df["GrLivArea"] < 4000].copy()
    X = df_model[["GrLivArea"]]
    y = df_model["SalePrice"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2 = st.columns(2)
    col1.metric("Model R² Score", f"{r2:.3f}")
    col2.metric("RMSE", f"${rmse:,.2f}")

    st.divider()

    # Interactive prediction
    st.markdown("#### Predict House Price")
    living_area = st.number_input(
        "Enter Above Ground Living Area (sq ft)",
        min_value=500, max_value=4000, value=1500
    )

    predicted_price = model.predict([[living_area]])[0]
    st.success(f"**Predicted Price: ${predicted_price:,.2f}**")

    # Plot fit line
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.scatter(X_test, y_test, alpha=0.6, color="#3498db")
    ax.plot(X_test, y_pred, color="#e74c3c", linewidth=2)
    ax.set_xlabel("Living Area")
    ax.set_ylabel("Sale Price")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------- Footer ----------------------
st.sidebar.divider()
st.sidebar.caption("ACC102 | Track4 | Interactive Data Tool")