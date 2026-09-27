import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from src.data_loader import load_nlp_data
from src.similarity_engine import SimilarityEngine


def main():

    df = load_nlp_data()

    engine = SimilarityEngine(df)

    complaint = """
    I was charged twice for the same purchase on my credit card.
    I contacted the company several times but the duplicate charge
    has still not been refunded.
    """

    print("\nFinding related complaints...\n")

    results = engine.find_similar(
        complaint
    )

    for i, result in enumerate(results, 1):

        print("=" * 70)

        print(f"Result #{i}")

        print(
            f"Complaint ID: {result['Complaint ID']}"
        )

        print(
            f"Product: {result['Product']}"
        )

        print(
            f"Issue: {result['Issue']}"
        )

        print(
            f"Company: {result['Company']}"
        )

        print(
            f"Similarity: "
            f"{result['Similarity'] * 100:.2f}%"
        )

        print(
            f"\n{result['Narrative'][:500]}"
        )


if __name__ == "__main__":
    main()