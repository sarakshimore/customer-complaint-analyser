import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from src.complaint_pipeline import ComplaintPipeline


def main():

    complaint = """
    I noticed that my credit card was charged twice for the
    same purchase. I contacted the company several times but
    the duplicate charge has still not been refunded.
    """

    print("\nLoading NLP pipeline...\n")

    pipeline = ComplaintPipeline()

    result = pipeline.analyze(
        complaint
    )

    print("=" * 60)

    print(
        "PRODUCT:",
        result["product"]["product"]
    )

    print(
        "CONFIDENCE:",
        result["product"]["confidence"]
    )

    print(
        "SENTIMENT:",
        result["sentiment"]["sentiment"]
    )

    print(
        "SENTIMENT CONFIDENCE:",
        result["sentiment"]["confidence"]
    )

    print(
        "SEVERITY:",
        result["severity"]["severity"]
    )

    print(
        "SEVERITY SCORE:",
        result["severity"]["score"]
    )

    print(
        "DEPARTMENT:",
        result["department"]
    )

    print(
        "\nSUMMARY:"
    )

    print(
        result["summary"]
    )

    print(
        "\nKEY ISSUES:"
    )

    for item in result["key_issues"]:

        print(
            "-",
            item["keyword"]
        )

    print("=" * 60)


if __name__ == "__main__":
    main()