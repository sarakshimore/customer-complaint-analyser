import pandas as pd
import os

INPUT_FILE = "complaints.csv"
OUTPUT_FILE = "cfpb_last_3_months.csv"

START_DATE = "2026-06-12"
END_DATE = "2026-09-12"

CHUNK_SIZE = 100_000

total_rows = 0
selected_rows = 0
first_chunk = True

print("Starting processing...")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Date range: {START_DATE} to {END_DATE}\n")

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    total_rows += len(chunk)

    # Convert date column
    chunk["Date received"] = pd.to_datetime(
        chunk["Date received"],
        errors="coerce"
    )

    # Select requested date range
    mask = (
        (chunk["Date received"] >= START_DATE) &
        (chunk["Date received"] <= END_DATE)
    )

    filtered = chunk.loc[mask]

    selected_rows += len(filtered)

    # Append to output file
    if not filtered.empty:
        filtered.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )
        first_chunk = False

    # Progress
    if total_rows % 1_000_000 == 0:
        print(
            f"Processed: {total_rows:,} | "
            f"Selected: {selected_rows:,}"
        )

print("\n==============================")
print("PROCESSING COMPLETE")
print("==============================")
print(f"Total rows scanned : {total_rows:,}")
print(f"Rows selected      : {selected_rows:,}")

if os.path.exists(OUTPUT_FILE):
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Output file size   : {size_mb:.2f} MB")
    print(f"Output file        : {OUTPUT_FILE}")
else:
    print("No output file was created.")