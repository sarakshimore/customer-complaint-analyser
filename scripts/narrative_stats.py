import pandas as pd

FILE = "cfpb_last_3_months.csv"
CHUNK_SIZE = 100_000

channel_narrative_counts = {}
product_narrative_counts = {}
issue_narrative_counts = {}

total_narratives = 0

print("Analyzing complaints with narratives...\n")

for chunk in pd.read_csv(
    FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    # Keep only records with narrative
    mask = (
        chunk["Consumer complaint narrative"].notna()
        &
        chunk["Consumer complaint narrative"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    df = chunk.loc[mask]

    total_narratives += len(df)

    # Channel
    for value, count in df["Submitted via"].value_counts().items():
        channel_narrative_counts[value] = (
            channel_narrative_counts.get(value, 0) + count
        )

    # Product
    for value, count in df["Product"].value_counts().items():
        product_narrative_counts[value] = (
            product_narrative_counts.get(value, 0) + count
        )

    # Issue
    for value, count in df["Issue"].value_counts().items():
        issue_narrative_counts[value] = (
            issue_narrative_counts.get(value, 0) + count
        )

print("=" * 60)
print("NARRATIVE DATASET")
print("=" * 60)

print(f"\nTotal narratives: {total_narratives:,}")

print("\n" + "=" * 60)
print("NARRATIVES BY CHANNEL")
print("=" * 60)

for channel, count in sorted(
    channel_narrative_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    percentage = count / total_narratives * 100
    print(f"{count:>8,}  {percentage:>6.2f}%  {channel}")

print("\n" + "=" * 60)
print("NARRATIVES BY PRODUCT")
print("=" * 60)

for product, count in sorted(
    product_narrative_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    percentage = count / total_narratives * 100
    print(f"{count:>8,}  {percentage:>6.2f}%  {product}")

print("\n" + "=" * 60)
print("TOP 30 ISSUES AMONG NARRATIVES")
print("=" * 60)

for issue, count in sorted(
    issue_narrative_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:30]:
    percentage = count / total_narratives * 100
    print(f"{count:>8,}  {percentage:>6.2f}%  {issue}")