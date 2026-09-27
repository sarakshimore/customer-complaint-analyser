import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import SIMILARITY_MODEL


def build_embeddings(
    csv_path,
    output_path=None,
    text_column="Consumer complaint narrative"
):
    """
    Generate Sentence-BERT embeddings for the supplied CSV dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.

    output_path : str or Path, optional
        Where to save the embeddings.
        If not provided, embeddings are saved next to the CSV.

    text_column : str
        Column containing the complaint narrative.

    Returns
    -------
    np.ndarray
        Generated embeddings.
    """

    import pandas as pd

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    print(f"Loading dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"Complaints: {len(df):,}")

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' was not found in the dataset.\n"
            f"Available columns: {list(df.columns)}"
        )

    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not any(texts):
        raise ValueError(
            "The complaint text column contains no usable text."
        )

    print("Loading Sentence-BERT...")

    model = SentenceTransformer(
        SIMILARITY_MODEL
    )

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings)

    # If no output path is supplied, create an embeddings
    # directory next to the input dataset.
    if output_path is None:
        output_dir = (
            csv_path.parent / "embeddings"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir /
            f"{csv_path.stem}_embeddings.npy"
        )

    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    np.save(
        output_path,
        embeddings
    )

    # Save metadata so we know exactly which dataset
    # produced these embeddings.
    metadata_path = output_path.with_suffix(
        ".meta.txt"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Dataset: {csv_path.resolve()}\n"
        )

        file.write(
            f"Rows: {len(df)}\n"
        )

        file.write(
            f"Text column: {text_column}\n"
        )

        file.write(
            f"Embedding shape: {embeddings.shape}\n"
        )

        file.write(
            f"Model: {SIMILARITY_MODEL}\n"
        )

    print(
        f"Saved embeddings to: {output_path}"
    )

    print(
        f"Saved metadata to: {metadata_path}"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    return embeddings


def main():

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python scripts/build_embeddings.py "
            "<csv_path> [output_path]"
        )

        sys.exit(1)

    csv_path = sys.argv[1]

    output_path = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else None
    )

    build_embeddings(
        csv_path=csv_path,
        output_path=output_path
    )


if __name__ == "__main__":
    main()