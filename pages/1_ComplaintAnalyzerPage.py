import streamlit as st
import pandas as pd

from src.complaint_pipeline import ComplaintPipeline


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Complaint Analyzer",
    layout="wide"
)


# ============================================================
# Load Models
# ============================================================

@st.cache_resource
def load_pipeline():
    return ComplaintPipeline()


pipeline = load_pipeline()
st.session_state["pipeline"] = pipeline


# ============================================================
# Page Header
# ============================================================

st.title("Customer Complaint Analyzer")

st.write(
    "Upload a CSV containing customer complaints to perform "
    "batch NLP analysis."
)


# ============================================================
# Upload Complaint Dataset
# ============================================================

st.subheader("Upload Complaint Dataset")

st.write(
    "Upload a CSV file containing customer complaints. "
    "The complaint text will be analyzed in the next step."
)


# ============================================================
# CSV Uploader
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# ============================================================
# Load New CSV
# ============================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        # Store uploaded dataset in session
        st.session_state["uploaded_df"] = df.copy()

        st.session_state["uploaded_file_name"] = (
            uploaded_file.name
        )

    except Exception as e:

        st.error(
            f"Could not read the CSV file: {e}"
        )

        st.stop()


# ============================================================
# Load Previously Uploaded CSV
# ============================================================

elif "uploaded_df" in st.session_state:

    df = st.session_state["uploaded_df"].copy()

    st.success(
        f"Previously uploaded dataset: "
        f"`{st.session_state.get('uploaded_file_name', 'CSV file')}`"
    )

else:

    df = None


# ============================================================
# Dataset Available
# ============================================================

if df is not None:

    st.success(
        f"Dataset loaded successfully: "
        f"{len(df):,} rows."
    )


    # ========================================================
    # Dataset Preview
    # ========================================================

    st.write("### Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # Select Complaint Column
    # ========================================================

    st.write(
        "### Select Complaint Column"
    )


    text_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()


    if not text_columns:

        st.error(
            "No text columns were found in the uploaded CSV."
        )

        st.stop()


    # ========================================================
    # Automatically Detect Complaint Column
    # ========================================================

    default_index = 0

    possible_names = [
        "complaint",
        "complaint_text",
        "consumer complaint narrative",
        "narrative",
        "text",
        "description",
        "message"
    ]


    for i, column in enumerate(text_columns):

        if column.lower() in possible_names:

            default_index = i
            break


    complaint_column = st.selectbox(
        "Which column contains the complaint text?",
        text_columns,
        index=default_index
    )


    st.info(
        f"Selected complaint column: "
        f"`{complaint_column}`"
    )


    # ========================================================
    # Analyze CSV
    # ========================================================

    if st.button(
        "Analyze Complaints",
        type="primary"
    ):

        with st.spinner(
            f"Analyzing {len(df):,} complaints..."
        ):

            results_df = pipeline.analyze_batch(
                df,
                complaint_column
            )


        # ----------------------------------------------------
        # Store Analysis Results
        # ----------------------------------------------------

        st.session_state["analysis_results"] = results_df


        st.success(
            f"Analysis completed for "
            f"{len(results_df):,} complaints."
        )


    # ========================================================
    # Show Analysis Available
    # ========================================================

    if "analysis_results" in st.session_state:

        results_df = st.session_state[
            "analysis_results"
        ]


        st.success(
            f"Analysis results available for "
            f"{len(results_df):,} complaints."
        )


        # ----------------------------------------------------
        # View Analytics
        # ----------------------------------------------------

        st.divider()

        if st.button(
            "View Analytics",
            type="primary"
        ):

            st.switch_page(
                "pages/2_AnalyticsPage.py"
            )