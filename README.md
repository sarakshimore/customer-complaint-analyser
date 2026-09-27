# NLP-Based Multi-Channel Customer Complaint Intelligence System

An NLP-based customer complaint intelligence system that automatically analyzes customer complaints and provides structured insights such as product classification, sentiment, severity, escalation detection, key issue extraction, summarization, department recommendation, semantic similarity, and complaint clustering.

The system is implemented using **Python** and **Streamlit** and combines multiple NLP and machine learning techniques to transform unstructured complaint text into actionable information.

---

## Overview

Customer complaint datasets contain large amounts of unstructured textual information that can be difficult to analyze manually.

This project provides an automated complaint analysis pipeline that can:

- Classify complaints into financial product categories
- Detect customer sentiment
- Assess complaint severity
- Detect potential escalation indicators
- Extract important complaint issues
- Generate complaint summaries
- Recommend the appropriate department
- Find semantically similar complaints
- Group semantically related complaints into clusters
- Provide complaint analytics and visualizations
- Export complete analysis results as a CSV file

The application allows users to upload a complaint dataset, select the complaint-text column, and perform batch analysis through an interactive Streamlit interface.

---

## Key Features

### 1. Complaint Product Classification

The system predicts the financial product associated with a complaint using:

- **TF-IDF**
- **Logistic Regression**

The classifier predicts categories such as:

- Checking or savings account
- Credit card
- Credit reporting
- Debt collection
- Debt/credit management
- Money transfer
- Mortgage
- Payday loan
- Prepaid card
- Student loan
- Vehicle loan

#### Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 81.85% |
| Macro Precision | 77.33% |
| Macro Recall | 73.70% |
| Macro F1 | 74.40% |
| Weighted F1 | 81.89% |

---

### 2. Sentiment Analysis

A pretrained RoBERTa-based sentiment model is used to identify the sentiment expressed in a complaint.

The system identifies:

- Positive
- Neutral
- Negative

It also provides the model confidence score.

**Model:**

`cardiffnlp/twitter-roberta-base-sentiment-latest`

---

### 3. Severity Assessment

Complaint severity is assessed using a **rule-based approach**.

The system considers indicators such as:

- Fraud
- Scams
- Identity theft
- Unauthorized transactions
- Legal action
- Court-related issues
- Foreclosure
- Bankruptcy
- Repeated unsuccessful resolution attempts
- Financial impact
- Duration of unresolved issues

Complaints are categorized as:

- Low
- Medium
- High
- Critical

> **Note:** Severity assessment is implemented using rule-based logic and is not a supervised machine learning classifier.

---

### 4. Escalation Detection

The system identifies complaints that may require additional attention.

Escalation indicators include:

- Fraud or unauthorized transactions
- Identity theft
- Legal action
- Repeated attempts to resolve an issue
- Lack of response
- Unresolved complaints
- References to managers or supervisors

The output is:

- Escalation Detected
- Not Detected

---

### 5. Key Issue Extraction

**YAKE (Yet Another Keyword Extractor)** is used to identify important phrases from complaint narratives.

The system extracts up to five relevant key phrases while filtering generic and redundant terms.

**Example:**

> "There are several unauthorized transactions on my credit card that I did not make."

Possible extracted issues:

- unauthorized transactions
- credit card
- transactions

---

### 6. Complaint Summarization

The system generates a concise summary of longer complaint narratives using **DistilBART CNN**.

**Model:**

`sshleifer/distilbart-cnn-12-6`

This allows users to understand the main complaint without reading the complete narrative.

---

### 7. Department Recommendation

Based on the predicted product category, the system recommends the department that should handle the complaint.

| Product | Recommended Department |
|---|---|
| Checking or savings account | Banking Operations |
| Credit card | Credit Card Operations |
| Credit reporting | Credit Reporting / Compliance |
| Debt collection | Collections Department |
| Debt/credit management | Credit Management |
| Money transfer | Payments / Money Transfer |
| Mortgage | Mortgage Operations |
| Payday loan | Lending Operations |
| Prepaid card | Prepaid Card Operations |
| Student loan | Student Loan Operations |
| Vehicle loan or lease | Vehicle Finance |

---

### 8. Semantic Complaint Similarity

The system uses **Sentence-BERT** to convert complaint narratives into semantic embeddings.

**Cosine similarity** is then used to identify complaints with similar meanings.

For example:

> "Someone made transactions on my credit card that I did not authorize."

and

> "There are purchases on my card that I never made."

may be identified as semantically similar even though the wording is different.

This enables users to discover related complaints based on meaning rather than exact keyword matches.

---

### 9. Semantic Complaint Clustering

Complaint embeddings are grouped using **K-Means clustering**.

The system evaluates different cluster counts and uses the **silhouette score** to select an appropriate number of clusters.

This helps identify recurring groups of semantically similar complaints.

For example, one cluster may contain complaints primarily related to:

- Unauthorized transactions
- Billing issues
- Payment problems
- Account access

> **Note:** Cluster labels require human interpretation. The clustering system identifies groups of semantically related complaints but does not automatically assign verified issue names to those groups.

---

### 10. Analytics Dashboard

The Analytics page provides insights into the complaint dataset.

The dashboard can display:

- Product distribution
- Issue distribution
- Company statistics
- State distribution
- Submission channels
- Daily complaint trends
- Monthly product trends
- Response statistics
- Sentiment distribution
- Severity distribution
- Escalation statistics
- Semantic cluster distribution
- Similar complaint results

The complete analysis results can also be downloaded as a CSV file.

---

## Dataset

The primary dataset used in this project is the **CFPB Consumer Complaint Database**.

### Source

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

[CFPB API Documentation](https://cfpb.github.io/api/ccdb/)

[CFPB API Field Reference](https://cfpb.github.io/api/ccdb/fields.html)

### Dataset Processing

The original CFPB dataset contained approximately **17.7 million complaints**.

The project filtered the dataset to complaints received between:

**June 12, 2026 – September 12, 2026**

This resulted in:

**1,895,573 complaints**

Only complaints containing consumer narrative text were used for NLP model development.

After filtering for available narratives:

**19,559 complaints**

were available for NLP processing.

The NLP dataset was divided into:

| Dataset | Records |
|---|---:|
| Training | 15,647 |
| Testing | 3,912 |
| Total | 19,559 |

### Important Dataset Note

The narrative subset used for NLP contains complaints submitted through the **Web** channel.

Therefore, the project does not claim that the NLP models were evaluated across all submission channels.

The `Submitted via` field is still used for structured complaint analytics.

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │   Complaint CSV     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Streamlit Input   │
                    │   & Column Select   │
                    └──────────┬──────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │    Complaint Pipeline    │
                  └────────────┬─────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   Product Classification  Sentiment          Severity &
   TF-IDF + Logistic       RoBERTa            Escalation
   Regression
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
     Key Issues          Summarization       Department
       YAKE               DistilBART         Recommendation
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Sentence-BERT       │
                    │ Semantic Embeddings │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Similarity Search       K-Means
             Cosine Similarity       Clustering
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Analysis Results &  │
                    │ Analytics Dashboard │
                    └─────────────────────┘
```

---

## Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Logistic Regression
- K-Means Clustering
- TF-IDF

### Natural Language Processing

- Hugging Face Transformers
- Sentence Transformers
- RoBERTa
- DistilBART
- YAKE

### Data Processing

- Pandas
- NumPy

### Application

- Streamlit

### Similarity Analysis

- Sentence-BERT
- Cosine Similarity

---

## Project Structure

```text
customer-complaint-analyser/
│
├── data/
│   ├── raw/
│   │   └── cfpb_last_3_months.csv
│   │
│   └── processed/
│       ├── analytics_summary.csv
│       ├── channel_stats.csv
│       ├── company_stats.csv
│       ├── daily_stats.csv
│       ├── dataset_profile.csv
│       ├── issue_stats.csv
│       ├── monthly_product_stats.csv
│       ├── nlp_dataset.csv
│       ├── product_stats.csv
│       ├── response_stats.csv
│       └── state_stats.csv
│
├── models/
│   ├── product_logistic_regression.pkl
│   ├── product_tfidf.pkl
│   ├── product_labels.pkl
│   ├── sentiment/
│   ├── embeddings/
│   └── distilbert_product_model/
│
├── pages/
│   ├── 1_ComplaintAnalyzerPage.py
│   └── 2_AnalyticsPage.py
│
├── scripts/
│   ├── build_embeddings.py
│   ├── create_analytics.py
│   ├── create_nlp_dataset.py
│   ├── dataset_stats.py
│   ├── filter_cfpb.py
│   ├── inspect_dataset.py
│   ├── narrative_stats.py
│   ├── predict_product.py
│   ├── profile_nlp_data.py
│   ├── test_pipeline.py
│   ├── test_severity.py
│   ├── test_similarity.py
│   └── train_product_classifier.py
│
├── src/
│   ├── complaint_clusterer.py
│   ├── complaint_pipeline.py
│   ├── config.py
│   ├── data_loader.py
│   ├── key_issue_extractor.py
│   ├── product_classifier.py
│   ├── routing.py
│   ├── sentiment_analyzer.py
│   ├── sentiment_interpreter.py
│   ├── severity_analyzer.py
│   ├── similarity_engine.py
│   └── summarizer.py
│
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
```

> **Note:** Large datasets, generated processed data, virtual environments, embeddings, and large pretrained model artifacts are excluded from version control using `.gitignore`.

---

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd customer-complaint-analyser
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

From the project root, run:

```bash
streamlit run app.py
```

The application will open in your default browser.

---

## Using the Application

### Step 1 — Upload Dataset

Open the **Complaint Analyzer** page and upload a CSV containing complaint narratives.

### Step 2 — Select Complaint Column

Select the column containing the complaint text.

The application supports common complaint-text column names such as:

- Complaint
- Complaint Text
- Consumer Complaint Narrative
- Narrative
- Text
- Description
- Message

### Step 3 — Analyze Complaints

Click:

**Analyze Complaints**

The system processes the complaints through the complete NLP pipeline.

### Step 4 — Review Results

The generated results include:

```text
Predicted Product
Product Confidence
Sentiment
Sentiment Confidence
Severity
Severity Score
Escalation Detected
Key Issues
Department
Cluster ID
```

### Step 5 — View Analytics

Click **View Analytics** to explore aggregated complaint statistics and semantic analysis.

### Step 6 — Export Results

The complete analysis table can be downloaded as a CSV file.

---

## Model Evaluation

The primary supervised classification model is **TF-IDF with Logistic Regression**.

### TF-IDF + Logistic Regression

```text
Accuracy        : 81.85%
Macro Precision : 77.33%
Macro Recall    : 73.70%
Macro F1        : 74.40%
Weighted F1     : 81.89%
```

A DistilBERT-based product classification model was also evaluated as an advanced comparison.

The final application uses the **TF-IDF + Logistic Regression** model because it provided slightly better macro F1 performance while being substantially faster and more lightweight for the application.

---

## Example

### Input

```text
I was charged twice for the same purchase on my credit card
and I want the extra charge refunded.
```

### Example Output

```text
Predicted Product:
Credit card

Product Confidence:
87.17%

Sentiment:
Negative

Severity:
Medium

Escalation:
Not Detected

Department:
Credit Card Operations
```

The system can additionally extract key issues, generate a summary, assign a semantic cluster, and identify similar complaints.

---

## Design Considerations

### Product Classification

Product classification is a supervised machine learning task trained using labeled CFPB complaint data.

### Sentiment Analysis

Sentiment analysis uses a pretrained transformer model.

### Severity and Escalation

Severity and escalation are implemented using rule-based logic rather than trained classification models.

### Semantic Analysis

Sentence-BERT provides pretrained semantic representations.

The project uses these embeddings for:

- Similarity search
- Complaint clustering

Sentence-BERT itself is not trained from scratch in this project.

---

## Limitations

- The NLP dataset contains only complaints with available narrative text.
- The narrative subset used for NLP contains Web-submitted complaints.
- Severity assessment is rule-based and does not have a machine learning accuracy metric.
- Escalation detection is based on predefined indicators.
- K-Means cluster labels require human interpretation.
- Semantic clustering identifies groups of semantically related complaints but does not automatically assign verified issue names to those groups.
- The system is primarily designed for English complaint narratives.
- Transformer-based components require additional computational resources compared with traditional NLP methods.

---

## Future Scope

Potential future extensions include:

- Multi-language complaint analysis
- Supervised severity classification using labeled data
- Supervised escalation prediction
- Advanced topic modeling
- Automatic cluster-label generation
- Explainable AI for model predictions
- Real-time complaint monitoring
- Complaint trend forecasting
- Integration with CRM and ticketing systems
- Automated response recommendation
- Human-in-the-loop complaint review
- Advanced transformer-based classification models

---

## Project Objective

The primary objective of this project is to demonstrate how **NLP, machine learning, semantic embeddings, and rule-based analysis** can be combined into a single application for automated customer complaint intelligence.

Instead of analyzing complaint text manually, the system converts unstructured complaints into structured information that can support:

- Complaint understanding
- Prioritization
- Routing
- Similar complaint discovery
- Recurring issue identification
- Complaint analytics

---

## Data Source

**Consumer Financial Protection Bureau (CFPB)**

Consumer Complaint Database:

https://www.consumerfinance.gov/data-research/consumer-complaints/

CFPB API Documentation:

https://cfpb.github.io/api/ccdb/

CFPB API Field Reference:

https://cfpb.github.io/api/ccdb/fields.html

---

## License

This project is developed for academic and educational purposes.

The CFPB Consumer Complaint Database is provided by the Consumer Financial Protection Bureau. Refer to the CFPB data source for applicable data usage terms and conditions.
