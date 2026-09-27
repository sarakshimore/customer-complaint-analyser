from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from .config import (
    SUMMARIZATION_MODEL,
    MAX_SUMMARY_LENGTH,
    MIN_SUMMARY_LENGTH
)


class ComplaintSummarizer:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            SUMMARIZATION_MODEL
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            SUMMARIZATION_MODEL
        )

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model.to(self.device)
        self.model.eval()

    def summarize(self, text):

        if not text or not text.strip():
            return ""

        text = text.strip()

        # Short complaints don't need summarization
        if len(text.split()) < 40:
            return text

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output_ids = self.model.generate(
                **inputs,
                max_length=MAX_SUMMARY_LENGTH,
                min_length=MIN_SUMMARY_LENGTH,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )