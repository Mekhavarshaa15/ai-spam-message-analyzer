"""Download, prepare, train, evaluate, and save the spam message classifier."""

from pathlib import Path
from io import BytesIO
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = DATA_DIR / "SMSSpamCollection.tsv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"


def get_dataset_path():
    """Use a local dataset when present, otherwise download it from UCI."""
    if DATASET_PATH.is_file():
        print(f"Using local dataset: {DATASET_PATH}")
        return DATASET_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading the UCI SMS Spam Collection dataset...")
    try:
        with urlopen(DATASET_URL, timeout=30) as response:
            archive_bytes = response.read()
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "Could not download the dataset. Check your internet connection, "
            "then try again. You can also place the UCI file named "
            "'SMSSpamCollection' in the data folder."
        ) from error

    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            member = "SMSSpamCollection"
            if member not in archive.namelist():
                raise RuntimeError(
                    "The downloaded dataset archive did not contain "
                    "'SMSSpamCollection'."
                )
            DATASET_PATH.write_bytes(archive.read(member))
    except BadZipFile as error:
        raise RuntimeError("The downloaded dataset was not a valid ZIP archive.") from error

    print(f"Dataset saved to: {DATASET_PATH}")
    return DATASET_PATH


def load_and_prepare_data(dataset_path):
    """Read the tab-separated labels and messages and remove unusable rows."""
    try:
        data = pd.read_csv(
            dataset_path,
            sep="\t",
            header=None,
            names=["label", "message"],
            usecols=[0, 1],
            # Latin-1 safely reads the original collection's accented characters.
            encoding="latin-1",
            on_bad_lines="error",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise RuntimeError(
            f"Could not read the SMS dataset at {dataset_path}: {error}"
        ) from error

    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].fillna("").astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"])]
    data = data[data["message"] != ""].drop_duplicates(subset=["label", "message"])
    data["label"] = data["label"].map({"ham": "not spam", "spam": "spam"})

    if data.empty or data["label"].nunique() != 2:
        raise ValueError(
            "The dataset must contain non-empty messages for both ham and spam."
        )

    return data


def train_and_evaluate(data):
    """Fit the TF-IDF and Naive Bayes model, then report held-out test metrics."""
    features_train, features_test, labels_train, labels_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    train_vectors = vectorizer.fit_transform(features_train)
    test_vectors = vectorizer.transform(features_test)

    classifier = MultinomialNB()
    classifier.fit(train_vectors, labels_train)
    predictions = classifier.predict(test_vectors)

    print("\nEvaluation on the held-out test set")
    print(f"Accuracy:  {accuracy_score(labels_test, predictions):.4f}")
    print(
        "Precision (spam): "
        f"{precision_score(labels_test, predictions, pos_label='spam', zero_division=0):.4f}"
    )
    print(
        "Recall (spam):    "
        f"{recall_score(labels_test, predictions, pos_label='spam', zero_division=0):.4f}"
    )
    print(
        "F1-score (spam):  "
        f"{f1_score(labels_test, predictions, pos_label='spam', zero_division=0):.4f}"
    )
    print("\nClassification report:")
    print(classification_report(labels_test, predictions, zero_division=0))

    return classifier, vectorizer


def main():
    """Run the complete training workflow and save reusable model files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = get_dataset_path()
    data = load_and_prepare_data(dataset_path)
    print(f"Prepared {len(data)} labeled messages.")
    print(f"Class counts:\n{data['label'].value_counts().to_string()}")

    classifier, vectorizer = train_and_evaluate(data)
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved classifier to: {MODEL_PATH}")
    print(f"Saved TF-IDF vectorizer to: {VECTORIZER_PATH}")
    print("Training complete. Start the app with: streamlit run app.py")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Training failed: {error}") from error
