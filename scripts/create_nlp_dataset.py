import pandas as pd
import os

INPUT_FILE = "data/raw/cfpb_last_3_months.csv"
OUTPUT_FILE = "data/processed/nlp_dataset.csv"

CHUNK_SIZE = 100_000

COLUMNS = [
    "Complaint ID",
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Submitted via",
    "Company response to consumer",
    "Timely response?"
]

total_rows = 0
narrative_rows = 0
first_chunk = True

os.makedirs("data/processed", exist_ok=True)

print("Creating NLP dataset...\n")

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    usecols=COLUMNS,
    low_memory=False
):
    total_rows += len(chunk)

    mask = (
        chunk["Consumer complaint narrative"].notna()
        &
        chunk["Consumer complaint narrative"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    filtered = chunk.loc[mask].copy()

    narrative_rows += len(filtered)

    if not filtered.empty:
        filtered.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    if total_rows % 1_000_000 == 0:
        print(
            f"Processed: {total_rows:,} | "
            f"Narratives: {narrative_rows:,}"
        )

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"Total records scanned : {total_rows:,}")
print(f"Narrative records     : {narrative_rows:,}")
print(f"Output                : {OUTPUT_FILE}")