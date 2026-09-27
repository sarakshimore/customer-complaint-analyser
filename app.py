import streamlit as st


st.set_page_config(
    page_title="Customer Complaint Intelligence",
    page_icon="",
    layout="wide"
)


st.title(
    "NLP-Based Multi-Channel "
    "Customer Complaint Intelligence System"
)


st.markdown(
    """
    ### Intelligent Complaint Analysis

    This system uses Natural Language Processing and Machine Learning
    to analyze customer complaints and extract actionable insights.

    #### Core Capabilities

    **Complaint Classification**
    
    Automatically identifies the financial product associated
    with a complaint.

    **Sentiment Analysis**
    
    Determines whether the complaint expresses positive,
    neutral, or negative sentiment.

    **Severity Assessment**
    
    Estimates complaint priority using interpretable
    linguistic indicators.

    **Key Issue Extraction**
    
    Identifies important phrases and issues within the complaint.

    **Semantic Complaint Detection**
    
    Finds complaints with similar meanings using
    Sentence-BERT embeddings.

    **Complaint Summarization**
    
    Generates a concise summary of lengthy customer narratives.

    **Complaint Analytics**
    
    Explores complaint trends, products, companies,
    issues, channels, and geographic distribution.
    """
)

st.divider()

st.info(
    "Use the sidebar to open the Complaint Analyzer "
    "or explore the dataset analytics."
)