import pandas as pd

FILE = "cfpb_last_3_months.csv"

print("Reading dataset sample...\n")

df = pd.read_csv(FILE, nrows=10_000, low_memory=False)

print("=" * 60)
print("SHAPE OF SAMPLE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

for i, column in enumerate(df.columns, 1):
    print(f"{i}. {column}")

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
missing_percentage = (missing / len(df)) * 100

missing_info = pd.DataFrame({
    "Missing": missing,
    "Percentage": missing_percentage.round(2)
})

print(missing_info.sort_values("Missing", ascending=False))

print("\n" + "=" * 60)
print("SAMPLE RECORDS")
print("=" * 60)

pd.set_option("display.max_columns", None)
print(df.head().to_string())