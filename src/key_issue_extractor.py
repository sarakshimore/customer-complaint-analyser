import re
import yake


STOP_PHRASES = {
    "credit",
    "card",
    "purchase",
    "noticed",
    "company",
    "customer",
    "account",
    "payment",
    "money",
    "amount",
    "charge",
    "charged",
    "issue",
    "problem",
    "help",
    "thing",
    "time",
    "times",
}


class KeyIssueExtractor:

    def __init__(self):

        self.extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.7,
            top=15,
            features=None
        )

    def clean_phrase(self, phrase):

        phrase = phrase.lower().strip()

        phrase = re.sub(
            r"[^\w\s-]",
            "",
            phrase
        )

        phrase = re.sub(
            r"\s+",
            " ",
            phrase
        )

        return phrase

    def is_valid_phrase(self, phrase):

        if not phrase:
            return False

        if phrase in STOP_PHRASES:
            return False

        words = phrase.split()

        # Remove single generic words
        if len(words) == 1:

            if len(phrase) < 5:
                return False

            if phrase in STOP_PHRASES:
                return False

        # Remove phrases made entirely from generic words
        if all(word in STOP_PHRASES for word in words):
            return False

        return True

    def is_redundant(self, phrase, selected):

        phrase_words = set(phrase.split())

        for existing in selected:

            existing_words = set(
                existing["keyword"].split()
            )

            # Exact duplicate
            if phrase == existing["keyword"]:
                return True

            # Phrase is completely contained in an existing phrase
            if phrase_words.issubset(existing_words):
                return True

            # Existing phrase is completely contained in this phrase
            if existing_words.issubset(phrase_words):
                return True

        return False

    def extract(self, text):

        keywords = self.extractor.extract_keywords(text)

        results = []

        for phrase, score in keywords:

            phrase = self.clean_phrase(phrase)

            if not self.is_valid_phrase(phrase):
                continue

            if self.is_redundant(
                phrase,
                results
            ):
                continue

            results.append({
                "keyword": phrase,
                "score": float(score)
            })

            if len(results) >= 5:
                break

        return results