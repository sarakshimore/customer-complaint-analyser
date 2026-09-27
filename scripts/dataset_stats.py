import pandas as pd

FILE = "cfpb_last_3_months.csv"
CHUNK_SIZE = 100_000

total_rows = 0
narrative_rows = 0

product_counts = {}
issue_counts = {}
channel_counts = {}

print("Scanning full dataset...\n")

for chunk in pd.read_csv(
    FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    total_rows += len(chunk)

    # Narrative availability
    narrative_mask = (
        chunk["Consumer complaint narrative"]
        .notna()
        &
        chunk["Consumer complaint narrative"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    narrative_rows += narrative_mask.sum()

    # Product distribution
    for value, count in chunk["Product"].value_counts().items():
        product_counts[value] = product_counts.get(value, 0) + count

    # Issue distribution
    for value, count in chunk["Issue"].value_counts().items():
        issue_counts[value] = issue_counts.get(value, 0) + count

    # Channel distribution
    for value, count in chunk["Submitted via"].value_counts().items():
        channel_counts[value] = channel_counts.get(value, 0) + count

    if total_rows % 1_000_000 == 0:
        print(f"Processed: {total_rows:,}")

print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(f"\nTotal records: {total_rows:,}")
print(f"Records with narrative: {narrative_rows:,}")
print(
    f"Narrative percentage: "
    f"{narrative_rows / total_rows * 100:.2f}%"
)

print("\n" + "=" * 60)
print("PRODUCT DISTRIBUTION")
print("=" * 60)

for product, count in sorted(
    product_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{count:>10,}  {product}")

print("\n" + "=" * 60)
print("CHANNEL DISTRIBUTION")
print("=" * 60)

for channel, count in sorted(
    channel_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{count:>10,}  {channel}")

print("\n" + "=" * 60)
print("TOP 20 ISSUES")
print("=" * 60)

for issue, count in sorted(
    issue_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]:
    print(f"{count:>10,}  {issue}")