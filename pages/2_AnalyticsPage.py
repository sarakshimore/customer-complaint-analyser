import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Complaint Analytics",
    layout="wide"
)


# ============================================================
# Page Header
# ============================================================

st.title("Complaint Analytics")

st.caption(
    "Analysis of the complaints processed during the current session."
)


# ============================================================
# Check for Analyzed Complaints
# ============================================================

if "analysis_results" not in st.session_state:

    st.info(
        "No analyzed complaints available. "
        "Go to Complaint Analyzer and analyze a CSV first."
    )

    st.stop()


results_df = st.session_state["analysis_results"].copy()

total_complaints = len(results_df)


# ============================================================
# Complaint Overview
# ============================================================

st.subheader("Complaint Overview")


# ============================================================
# KPI Cards
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Complaints",
        f"{total_complaints:,}"
    )


with col2:

    if "Sentiment" in results_df.columns:

        negative_count = (
            results_df["Sentiment"]
            .astype(str)
            .str.lower()
            .eq("negative")
            .sum()
        )

        st.metric(
            "Negative Complaints",
            f"{negative_count:,}"
        )

    else:

        st.metric(
            "Negative Complaints",
            "N/A"
        )


with col3:

    if "Escalation Detected" in results_df.columns:

        escalation_count = (
            results_df["Escalation Detected"]
            .eq(True)
            .sum()
        )

        st.metric(
            "Escalations",
            f"{escalation_count:,}"
        )

    else:

        st.metric(
            "Escalations",
            "N/A"
        )


with col4:

    if "Severity" in results_df.columns:

        critical_count = (
            results_df["Severity"]
            .astype(str)
            .str.lower()
            .eq("critical")
            .sum()
        )

        st.metric(
            "Critical Complaints",
            f"{critical_count:,}"
        )

    else:

        st.metric(
            "Critical Complaints",
            "N/A"
        )


st.divider()


# ============================================================
# Date Information
# ============================================================

if "Date Received" in results_df.columns:

    results_df["Date Received"] = pd.to_datetime(
        results_df["Date Received"],
        errors="coerce"
    )

    date_data = results_df.dropna(
        subset=["Date Received"]
    )

else:

    date_data = pd.DataFrame()


# ============================================================
# Complaint Volume Over Time
# ============================================================

st.subheader("Complaint Volume Over Time")


if not date_data.empty:

    daily = (
        date_data
        .groupby(
            date_data["Date Received"].dt.date
        )
        .size()
        .reset_index(name="Complaints")
    )

    daily["Date Received"] = pd.to_datetime(
        daily["Date Received"]
    )

    min_date = daily["Date Received"].min().date()
    max_date = daily["Date Received"].max().date()


    if min_date != max_date:

        date_range = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:

            start_date, end_date = date_range

            filtered_daily = daily[
                (daily["Date Received"].dt.date >= start_date)
                &
                (daily["Date Received"].dt.date <= end_date)
            ]

        else:

            filtered_daily = daily

    else:

        filtered_daily = daily


    fig = px.line(
        filtered_daily,
        x="Date Received",
        y="Complaints",
        markers=True,
        title="Daily Complaint Volume"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


else:

    st.info(
        "Date trend analysis is unavailable because the "
        "analyzed results do not contain a 'Date Received' column."
    )


# ============================================================
# Product Analysis
# ============================================================

if "Predicted Product" in results_df.columns:

    st.subheader("Product Complaint Analysis")

    product_column = "Predicted Product"

elif "Product" in results_df.columns:

    st.subheader("Product Complaint Analysis")

    product_column = "Product"

else:

    product_column = None


if product_column:

    product_counts = (
        results_df[product_column]
        .value_counts()
        .reset_index()
    )

    product_counts.columns = [
        "Product",
        "Complaints"
    ]

    fig = px.bar(
        product_counts,
        x="Complaints",
        y="Product",
        orientation="h",
        title="Complaint Volume by Product"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# Sentiment Analysis
# ============================================================

if "Sentiment" in results_df.columns:

    st.subheader("Sentiment Analysis")

    sentiment_counts = (
        results_df["Sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Complaints"
    ]

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Complaints",
        title="Complaints by Sentiment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# Severity Analysis
# ============================================================

if "Severity" in results_df.columns:

    st.subheader("Severity Analysis")

    severity_counts = (
        results_df["Severity"]
        .value_counts()
        .reindex(
            [
                "Low",
                "Medium",
                "High",
                "Critical"
            ],
            fill_value=0
        )
        .reset_index()
    )

    severity_counts.columns = [
        "Severity",
        "Complaints"
    ]

    fig = px.bar(
        severity_counts,
        x="Severity",
        y="Complaints",
        title="Complaints by Severity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# Escalation Analysis
# ============================================================

if "Escalation Detected" in results_df.columns:

    st.subheader("Escalation Analysis")

    escalation_counts = (
        results_df["Escalation Detected"]
        .map({
            True: "Escalation Required",
            False: "Normal Handling"
        })
        .value_counts()
        .reset_index()
    )

    escalation_counts.columns = [
        "Status",
        "Complaints"
    ]

    fig = px.pie(
        escalation_counts,
        names="Status",
        values="Complaints",
        hole=0.4,
        title="Escalation Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# Department Analysis
# ============================================================

if "Department" in results_df.columns:

    st.subheader(
        "Recommended Department Analysis"
    )

    department_counts = (
        results_df["Department"]
        .value_counts()
        .reset_index()
    )

    department_counts.columns = [
        "Department",
        "Complaints"
    ]

    fig = px.bar(
        department_counts,
        x="Complaints",
        y="Department",
        orientation="h",
        title="Complaints by Recommended Department"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# Submission Channels
# ============================================================

if "Submitted via" in results_df.columns:

    st.subheader(
        "Complaint Submission Channels"
    )

    channel_counts = (
        results_df["Submitted via"]
        .value_counts()
        .reset_index()
    )

    channel_counts.columns = [
        "Channel",
        "Complaints"
    ]

    fig = px.pie(
        channel_counts,
        names="Channel",
        values="Complaints",
        hole=0.4,
        title="How Complaints Were Submitted"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# Top Companies
# ============================================================

if "Company" in results_df.columns:

    st.subheader(
        "Companies with Most Complaints"
    )

    company_counts = (
        results_df["Company"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    company_counts.columns = [
        "Company",
        "Complaints"
    ]

    fig = px.bar(
        company_counts,
        x="Complaints",
        y="Company",
        orientation="h",
        title="Top Companies by Complaint Volume"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# State Analysis
# ============================================================

if "State" in results_df.columns:

    st.subheader(
        "Complaints by State"
    )

    state_counts = (
        results_df["State"]
        .value_counts()
        .head(20)
        .reset_index()
    )

    state_counts.columns = [
        "State",
        "Complaints"
    ]

    fig = px.bar(
        state_counts,
        x="State",
        y="Complaints",
        title="Top States by Complaint Volume"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# Semantic Complaint Analysis
# ============================================================

if "Cluster ID" in results_df.columns:

    st.divider()

    st.subheader(
        "Semantic Complaint Analysis"
    )

    st.caption(
        "Sentence-BERT embeddings are used to identify "
        "semantically related complaints and group them "
        "into complaint clusters."
    )

    # Remove missing cluster IDs
    cluster_data = results_df[
        results_df["Cluster ID"].notna()
    ].copy()

    if not cluster_data.empty:

        cluster_count = (
            cluster_data["Cluster ID"]
            .nunique()
        )

        # --------------------------------------------
        # Cluster KPI cards
        # --------------------------------------------

        cluster_col1, cluster_col2 = st.columns(2)

        with cluster_col1:

            st.metric(
                "Semantic Clusters",
                f"{cluster_count:,}"
            )

        with cluster_col2:

            # Try to retrieve silhouette score
            # from the pipeline stored in session state.
            clustering_score = None

            if "pipeline" in st.session_state:

                try:

                    clustering_score = (
                        st.session_state[
                            "pipeline"
                        ].get_clustering_score()
                    )

                except Exception:
                    clustering_score = None

            if clustering_score is not None:

                st.metric(
                    "Silhouette Score",
                    f"{clustering_score:.3f}"
                )

            else:

                st.metric(
                    "Silhouette Score",
                    "N/A"
                )

        st.write("### Complaint Cluster Distribution")

        cluster_counts = (
            cluster_data["Cluster ID"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        cluster_counts.columns = [
            "Cluster",
            "Complaints"
        ]

        cluster_counts["Cluster"] = (
            cluster_counts["Cluster"]
            .astype(int)
            .astype(str)
            .apply(lambda x: f"Cluster {x}")
        )

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Complaints",
            title="Complaints by Semantic Cluster"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------------------
        # Cluster exploration
        # --------------------------------------------

        st.write("### Explore Complaint Cluster")

        available_clusters = sorted(
            cluster_data["Cluster ID"]
            .unique()
            .tolist()
        )

        selected_cluster = st.selectbox(
            "Select a cluster",
            available_clusters,
            format_func=lambda x: (
                f"Cluster {int(x)}"
            )
        )

        selected_cluster_data = (
            cluster_data[
                cluster_data["Cluster ID"]
                == selected_cluster
            ]
            .copy()
        )

        st.write(
            f"**Cluster {int(selected_cluster)}** "
            f"contains "
            f"**{len(selected_cluster_data):,} "
            f"complaint(s).**"
        )

        display_columns = []

        for column in [
            "Complaint ID",
            "Complaint",
            "Predicted Product",
            "Sentiment",
            "Severity",
            "Company",
            "State",
            "Key Issues"
        ]:

            if column in selected_cluster_data.columns:
                display_columns.append(column)

        if display_columns:

            st.dataframe(
                selected_cluster_data[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.dataframe(
                selected_cluster_data,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "No semantic clusters were generated."
        )


# ============================================================
# Similar Complaint Detection
# ============================================================

if (
    "Cluster ID" in results_df.columns
    and "Complaint" in results_df.columns
):

    st.write("### Find Similar Complaints")

    st.caption(
        "Select a complaint to find other complaints "
        "with similar semantic meaning."
    )

    complaint_indices = results_df.index.tolist()

    selected_index = st.selectbox(
        "Select a complaint",
        complaint_indices,
        format_func=lambda i: (
            f"Complaint {i + 1}: "
            f"{str(results_df.loc[i, 'Complaint'])[:100]}"
            + (
                "..."
                if len(
                    str(
                        results_df.loc[
                            i,
                            "Complaint"
                        ]
                    )
                ) > 100
                else ""
            )
        )
    )

    selected_complaint = str(
        results_df.loc[
            selected_index,
            "Complaint"
        ]
    )

    st.write("**Selected Complaint:**")

    st.info(
        selected_complaint
    )

    if st.button(
        "Find Similar Complaints",
        type="primary"
    ):

        if "pipeline" not in st.session_state:

            st.error(
                "Similarity engine is not available "
                "for this analysis session."
            )

        else:

            try:

                similar_complaints = (
                    st.session_state[
                        "pipeline"
                    ].find_similar_complaints(
                        selected_complaint,
                        max_results=5,
                        similarity_threshold=0.40
                    )
                )

                # Remove the selected complaint itself
                # if it appears as the highest similarity result.
                filtered_results = []

                selected_id = None

                if "Complaint ID" in results_df.columns:

                    selected_id = results_df.loc[
                        selected_index,
                        "Complaint ID"
                    ]

                for item in similar_complaints:

                    if (
                        selected_id is not None
                        and item["Complaint ID"]
                        == selected_id
                    ):
                        continue

                    filtered_results.append(item)

                if filtered_results:

                    similarity_df = pd.DataFrame(
                        filtered_results
                    )

                    similarity_df[
                        "Similarity"
                    ] = (
                        similarity_df[
                            "Similarity"
                        ]
                        .astype(float)
                        .mul(100)
                        .round(2)
                        .astype(str)
                        + "%"
                    )

                    st.dataframe(
                        similarity_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No sufficiently similar complaints "
                        "were found."
                    )

            except Exception as e:

                st.error(
                    f"Could not find similar complaints: {e}"
                )


# ============================================================
# Analysis Results Table
# ============================================================

st.divider()

st.subheader("Analysis Results")

st.write(
    f"{total_complaints:,} analyzed complaint(s) available."
)

st.dataframe(
    results_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# Download Analysis Results
# ============================================================

csv_data = results_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Analysis Results",
    data=csv_data,
    file_name="complaint_analysis_results.csv",
    mime="text/csv"
)