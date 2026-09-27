import sys
from pathlib import Path

import pandas as pd


# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from src.config import RAW_DATA_PATH, BASE_DIR


OUTPUT_DIR = BASE_DIR / "data" / "processed"


def main():

    print("Loading CFPB dataset...")

    df = pd.read_csv(
        RAW_DATA_PATH,
        usecols=[
            "Date received",
            "Product",
            "Issue",
            "Company",
            "State",
            "Submitted via",
            "Timely response?",
            "Company response to consumer"
        ]
    )

    print(
        f"Loaded {len(df):,} complaints."
    )


    # =========================================================
    # Date processing
    # =========================================================

    df["Date received"] = pd.to_datetime(
        df["Date received"],
        errors="coerce"
    )


    # =========================================================
    # Product statistics
    # =========================================================

    product_stats = (
        df["Product"]
        .value_counts()
        .rename_axis("Product")
        .reset_index(name="Complaints")
    )

    product_stats.to_csv(
        OUTPUT_DIR / "product_stats.csv",
        index=False
    )


    # =========================================================
    # Issue statistics
    # =========================================================

    issue_stats = (
        df["Issue"]
        .value_counts()
        .rename_axis("Issue")
        .reset_index(name="Complaints")
    )

    issue_stats.to_csv(
        OUTPUT_DIR / "issue_stats.csv",
        index=False
    )


    # =========================================================
    # Company statistics
    # =========================================================

    company_stats = (
        df["Company"]
        .value_counts()
        .rename_axis("Company")
        .reset_index(name="Complaints")
    )

    company_stats.to_csv(
        OUTPUT_DIR / "company_stats.csv",
        index=False
    )


    # =========================================================
    # Channel statistics
    # =========================================================

    channel_stats = (
        df["Submitted via"]
        .value_counts()
        .rename_axis("Channel")
        .reset_index(name="Complaints")
    )

    channel_stats.to_csv(
        OUTPUT_DIR / "channel_stats.csv",
        index=False
    )


    # =========================================================
    # State statistics
    # =========================================================

    state_stats = (
        df["State"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("State")
        .reset_index(name="Complaints")
    )

    state_stats.to_csv(
        OUTPUT_DIR / "state_stats.csv",
        index=False
    )


    # =========================================================
    # Daily trend
    # =========================================================

    daily_stats = (
        df.dropna(subset=["Date received"])
        .groupby(
            df["Date received"].dt.date
        )
        .size()
        .reset_index(name="Complaints")
    )

    daily_stats.columns = [
        "Date",
        "Complaints"
    ]

    daily_stats["Date"] = pd.to_datetime(
        daily_stats["Date"]
    )

    daily_stats.to_csv(
        OUTPUT_DIR / "daily_stats.csv",
        index=False
    )


    # =========================================================
    # Product × Month trend
    # =========================================================

    df["Month"] = (
        df["Date received"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_product_stats = (
        df.groupby(
            ["Month", "Product"]
        )
        .size()
        .reset_index(name="Complaints")
    )

    monthly_product_stats.to_csv(
        OUTPUT_DIR / "monthly_product_stats.csv",
        index=False
    )


    # =========================================================
    # Response statistics
    # =========================================================

    response_stats = (
        df["Timely response?"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Timely Response")
        .reset_index(name="Complaints")
    )

    response_stats.to_csv(
        OUTPUT_DIR / "response_stats.csv",
        index=False
    )


    # =========================================================
    # Summary
    # =========================================================

    summary = pd.DataFrame({
        "Metric": [
            "Total Complaints",
            "Unique Products",
            "Unique Issues",
            "Unique Companies",
            "Unique States",
            "Submission Channels"
        ],
        "Value": [
            len(df),
            df["Product"].nunique(),
            df["Issue"].nunique(),
            df["Company"].nunique(),
            df["State"].nunique(),
            df["Submitted via"].nunique()
        ]
    })

    summary.to_csv(
        OUTPUT_DIR / "analytics_summary.csv",
        index=False
    )


    print("\nAnalytics files created successfully.")

    for file in OUTPUT_DIR.glob("*_stats.csv"):
        print(" -", file.name)

    print(" - analytics_summary.csv")


if __name__ == "__main__":
    main()