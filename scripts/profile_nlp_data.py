import pandas as pd
import os
from collections import Counter

FILE = "data/raw/cfpb_last_3_months.csv"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 100_000

# Counters
total_records = 0
narrative_records = 0

product_counts = Counter()
issue_counts = Counter()
subproduct_counts = Counter()
company_counts = Counter()

text_lengths = []

print("Profiling CFPB narrative dataset...\n")

for chunk in pd.read_csv(
    FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    total_records += len(chunk)

    # Keep records with usable narrative
    mask = (
        chunk["Consumer complaint narrative"].notna()
        &
        chunk["Consumer complaint narrative"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    df = chunk.loc[mask].copy()

    narrative_records += len(df)

    # Category counts
    product_counts.update(df["Product"].dropna())
    issue_counts.update(df["Issue"].dropna())
    subproduct_counts.update(df["Sub-product"].dropna())
    company_counts.update(df["Company"].dropna())

    # Text length
    lengths = (
        df["Consumer complaint narrative"]
        .astype(str)
        .str.split()
        .str.len()
        .tolist()
    )

    text_lengths.extend(lengths)

    if total_records % 1_000_000 == 0:
        print(f"Processed: {total_records:,}")

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total records              : {total_records:,}")
print(f"Records with narrative     : {narrative_records:,}")
print(
    f"Narrative percentage       : "
    f"{narrative_records / total_records * 100:.2f}%"
)

# --------------------------------------------------
# TEXT STATISTICS
# --------------------------------------------------

text_series = pd.Series(text_lengths)

print("\n" + "=" * 70)
print("TEXT LENGTH STATISTICS (WORDS)")
print("=" * 70)

print(f"Minimum                    : {text_series.min()}")
print(f"Maximum                    : {text_series.max()}")
print(f"Mean                       : {text_series.mean():.2f}")
print(f"Median                     : {text_series.median():.2f}")
print(f"25th percentile            : {text_series.quantile(0.25):.2f}")
print(f"75th percentile            : {text_series.quantile(0.75):.2f}")
print(f"95th percentile            : {text_series.quantile(0.95):.2f}")

# --------------------------------------------------
# PRODUCT
# --------------------------------------------------

print("\n" + "=" * 70)
print("PRODUCT DISTRIBUTION")
print("=" * 70)

for name, count in product_counts.most_common():
    percentage = count / narrative_records * 100
    print(f"{count:>7,}  {percentage:>6.2f}%  {name}")

# --------------------------------------------------
# ISSUE
# --------------------------------------------------

print("\n" + "=" * 70)
print("ISSUE DISTRIBUTION")
print("=" * 70)

print(f"Unique issues: {len(issue_counts)}\n")

for name, count in issue_counts.most_common(30):
    percentage = count / narrative_records * 100
    print(f"{count:>7,}  {percentage:>6.2f}%  {name}")

# --------------------------------------------------
# SUB-PRODUCT
# --------------------------------------------------

print("\n" + "=" * 70)
print("SUB-PRODUCT DISTRIBUTION")
print("=" * 70)

for name, count in subproduct_counts.most_common():
    percentage = count / narrative_records * 100
    print(f"{count:>7,}  {percentage:>6.2f}%  {name}")

# --------------------------------------------------
# COMPANIES
# --------------------------------------------------

print("\n" + "=" * 70)
print("COMPANY STATISTICS")
print("=" * 70)

print(f"Unique companies: {len(company_counts)}")

print("\nTop 20 companies:")

for name, count in company_counts.most_common(20):
    print(f"{count:>7,}  {name}")

# --------------------------------------------------
# SAVE BASIC PROFILE
# --------------------------------------------------

profile = {
    "total_records": total_records,
    "narrative_records": narrative_records,
    "narrative_percentage": narrative_records / total_records * 100,
    "unique_products": len(product_counts),
    "unique_issues": len(issue_counts),
    "unique_subproducts": len(subproduct_counts),
    "unique_companies": len(company_counts),
    "text_min_words": int(text_series.min()),
    "text_max_words": int(text_series.max()),
    "text_mean_words": float(text_series.mean()),
    "text_median_words": float(text_series.median()),
}

pd.DataFrame([profile]).to_csv(
    f"{OUTPUT_DIR}/dataset_profile.csv",
    index=False
)

print("\nProfile saved to:")
print(f"{OUTPUT_DIR}/dataset_profile.csv")