import pandas as pd

from .config import RAW_DATA_PATH, NLP_DATA_PATH


def load_nlp_data():
    """
    Load the 19,559 narrative-bearing complaints.
    """

    df = pd.read_csv(NLP_DATA_PATH)

    df["Consumer complaint narrative"] = (
        df["Consumer complaint narrative"]
        .fillna("")
        .astype(str)
    )

    return df


def load_structured_data():
    """
    Load the complete 3-month CFPB dataset.

    This is the 1.895M-record dataset and is used
    for dashboard-level structured analytics.
    """

    return pd.read_csv(RAW_DATA_PATH)


def get_product_distribution(df):
    return (
        df["Product"]
        .value_counts()
        .reset_index()
    )


def get_channel_distribution(df):
    return (
        df["Submitted via"]
        .value_counts()
        .reset_index()
    )


def get_issue_distribution(df):
    return (
        df["Issue"]
        .value_counts()
        .head(15)
        .reset_index()
    )


def get_company_distribution(df):
    return (
        df["Company"]
        .value_counts()
        .head(15)
        .reset_index()
    )