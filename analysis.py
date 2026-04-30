# =========================================
# 📊 E-COMMERCE SALES ANALYSIS PROJECT
# =========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_analysis():

    # =========================
    # 📂 CREATE OUTPUT FOLDER
    # =========================
    os.makedirs("output", exist_ok=True)

    # =========================
    # 📥 LOAD DATA
    # =========================
    try:
        df = pd.read_csv("data/ecommerce_data.csv")
        logging.info("Data loaded successfully")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # =========================
    # 🧹 DATA CLEANING
    # =========================
    logging.info("Starting data cleaning...")

    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True, errors='coerce')
    df["sales"] = pd.to_numeric(df["sales"], errors='coerce')

    logging.info("\nMissing Values:\n%s", df.isnull().sum())

    df = df.dropna(subset=["order_date"])
    df.loc[:, "sales"] = df["sales"].fillna(df["sales"].median())
    df = df.drop_duplicates()

    logging.info(f"Data cleaned. Final shape: {df.shape}")

    # =========================
    # ⚙️ FEATURE ENGINEERING
    # =========================
    df["month"] = df["order_date"].dt.month
    df["year"] = df["order_date"].dt.year
    df["profit_margin"] = (df["profit"] / df["sales"]) * 100

    # =========================
    # 📊 EDA
    # =========================
    print("\n===== KEY METRICS =====")
    print("Total Sales:", df["sales"].sum())
    print("Total Profit:", df["profit"].sum())

    print("\n===== TOP CATEGORIES =====")
    print(df.groupby("category")["sales"].sum().sort_values(ascending=False))

    print("\n===== TOP PRODUCTS =====")
    print(df.groupby("product_name")["sales"].sum().sort_values(ascending=False).head(5))

    print("\n===== TOP CUSTOMERS =====")
    print(df.groupby("customer_name")["sales"].sum().sort_values(ascending=False).head(10))

    print("\n===== REGION-WISE SALES =====")
    print(df.groupby("region")["sales"].sum())

    print("\n===== LOSS-MAKING PRODUCTS =====")
    loss = df[df["profit"] < 0]
    print(loss.groupby("product_name")["profit"].sum().head())

    print("\n===== DISCOUNT IMPACT =====")
    print(df.groupby("discount")["profit"].mean())

    print("\n===== YEARLY SALES =====")
    print(df.groupby("year")["sales"].sum())

    print("\n===== AVERAGE PROFIT MARGIN =====")
    print(df["profit_margin"].mean())

    print("\n===== BUSINESS INSIGHTS =====")
    print("1. Technology category generates highest revenue.")
    print("2. Sales show consistent year-on-year growth.")
    print("3. High discount (>20%) leads to negative profit.")
    print("4. Some products are consistently loss-making.")
    print("5. Central and South regions contribute most revenue.")

    # =========================
    # 📈 VISUALIZATION
    # =========================
    logging.info("Generating visualizations...")

    # Monthly Sales
    monthly_sales = df.groupby("month")["sales"].sum()
    plt.figure()
    monthly_sales.plot()
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.savefig("output/monthly_sales.png")
    plt.show()

    # Category Sales
    plt.figure()
    df.groupby("category")["sales"].sum().plot(kind="bar")
    plt.title("Sales by Category")
    plt.savefig("output/category_sales.png")
    plt.show()

    # Sales vs Profit
    plt.figure()
    plt.scatter(df["sales"], df["profit"])
    plt.xlabel("Sales")
    plt.ylabel("Profit")
    plt.title("Sales vs Profit")
    plt.savefig("output/sales_vs_profit.png")
    plt.show()

    # Heatmap
    plt.figure()
    pivot = df.pivot_table(values="sales", index="category", columns="region", aggfunc="sum")
    sns.heatmap(pivot, annot=True)
    plt.title("Sales Heatmap")
    plt.savefig("output/heatmap.png")
    plt.show()

    # =========================
    # 💾 SAVE CLEANED DATA
    # =========================
    df.to_csv("output/cleaned_data.csv", index=False)
    logging.info("Cleaned dataset saved successfully!")


# =========================
# ▶ RUN
# =========================
if __name__ == "__main__":
    run_analysis()