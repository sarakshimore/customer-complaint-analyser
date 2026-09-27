class SentimentInterpreter:

    def interpret(
        self,
        sentiment,
        severity,
        key_issues,
        complaint_text,
        escalation_detected=False
    ):

        sentiment = sentiment.lower()
        severity = severity.lower()

        # Convert key issues into readable text
        if key_issues:
            if isinstance(key_issues, list):
                issues = ", ".join(
                    item["keyword"]
                    if isinstance(item, dict)
                    else str(item)
                    for item in key_issues
                )
            else:
                issues = str(key_issues)

        else:
            issues = "the reported issue"

        # --------------------------------------------
        # Critical complaints
        # --------------------------------------------

        if severity == "critical":

            return (
                f"The customer expresses {sentiment} sentiment regarding "
                f"{issues}. The complaint has been classified as critical, "
                "indicating a potentially serious financial, security, "
                "or legal concern that may require immediate attention."
            )

        # --------------------------------------------
        # High severity + escalation
        # --------------------------------------------

        if severity == "high" and escalation_detected:

            return (
                f"The customer expresses {sentiment} sentiment regarding "
                f"{issues}. The complaint shows signs of escalation, "
                "including repeated or unsuccessful attempts to resolve "
                "the issue, and may require prompt attention."
            )

        # --------------------------------------------
        # High severity
        # --------------------------------------------

        if severity == "high":

            return (
                f"The customer expresses {sentiment} sentiment regarding "
                f"{issues}. The high severity indicates that the complaint "
                "may have a significant impact and should receive prompt "
                "attention."
            )

        # --------------------------------------------
        # Medium severity
        # --------------------------------------------

        if severity == "medium":

            return (
                f"The customer expresses {sentiment} sentiment regarding "
                f"{issues}. The complaint appears to concern a moderate-"
                "impact issue that requires resolution."
            )

        # --------------------------------------------
        # Positive sentiment
        # --------------------------------------------

        if sentiment == "positive":

            return (
                "The complaint contains positive sentiment, suggesting "
                "that the customer is satisfied with the handling or "
                "resolution of the issue."
            )

        # --------------------------------------------
        # Neutral sentiment
        # --------------------------------------------

        if sentiment == "neutral":

            return (
                f"The customer uses relatively neutral language while "
                f"reporting an issue related to {issues}."
            )

        # --------------------------------------------
        # Low severity negative sentiment
        # --------------------------------------------

        if sentiment == "negative":

            return (
                f"The customer expresses dissatisfaction regarding "
                f"{issues}, but the complaint does not indicate a "
                "high-impact issue."
            )

        # --------------------------------------------
        # Fallback
        # --------------------------------------------

        return (
            f"The complaint concerns {issues} and has been classified "
            f"with {sentiment} sentiment and {severity} severity."
        )