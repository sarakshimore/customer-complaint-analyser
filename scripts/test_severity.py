from src.severity_model import SeverityModel


model = SeverityModel()


test_complaints = [
    "I don't understand this fee on my account.",

    "I was charged twice for the same purchase and still haven't received my refund.",

    "I've contacted customer service 5 times and nobody has helped me resolve my problem.",

    "I was the victim of identity theft and this account wasn't mine.",

    "Someone made an unauthorized transaction using my bank account.",

    "Thank you for finally resolving my issue. I appreciate the help."
]


for i, complaint in enumerate(
    test_complaints,
    start=1
):

    result = model.predict(complaint)

    print("\n" + "=" * 60)

    print(f"Complaint {i}:")
    print(complaint)

    print("\nPredicted Severity:")
    print(result["severity"])

    print("\nConfidence:")
    print(f"{result['confidence']:.2%}")

    print("\nAll Scores:")

    for severity, score in result["scores"].items():

        print(
            f"{severity}: {score:.2%}"
        )