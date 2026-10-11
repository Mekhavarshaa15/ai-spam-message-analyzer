"""Streamlit interface for the AI Spam Message Analyzer."""

from pathlib import Path

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"


@st.cache_resource
def load_model_files():
    """Load the saved classifier and vectorizer once per Streamlit process."""
    if not MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(
            "The trained model files were not found. Run "
            "`python train_model.py` from the project folder first."
        )

    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return classifier, vectorizer


st.set_page_config(
    page_title="AI Spam Message Analyzer",
    page_icon="🛡️",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container { max-width: 850px; padding-top: 2.5rem; }
        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #eef4ff, #f4f0ff);
            border: 1px solid #e1e8f5;
            margin-bottom: 1.5rem;
        }
        .hero h1 { margin: 0 0 .4rem 0; color: #172554; }
        .hero p { color: #475569; margin: 0; }
        .result { padding: 1rem 1.2rem; border-radius: 12px; }
        .spam { background: #fff1f2; border: 1px solid #fecdd3; color: #9f1239; }
        .ham { background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; }
        @media (max-width: 600px) {
            .block-container { padding: 1rem; }
            .hero { padding: 1.2rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ AI Spam Message Analyzer</h1>
        <p>Check an SMS message with a machine-learning model trained on labeled messages.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Paste a message below to estimate whether it resembles spam or a legitimate SMS."
)

try:
    classifier, vectorizer = load_model_files()
except (FileNotFoundError, OSError, ValueError, EOFError) as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info("Open a terminal in the project folder, then run `python train_model.py`.")
    st.stop()

message = st.text_area(
    "Message to analyze",
    placeholder="Example: Congratulations! You have won a free prize. Call now!",
    height=150,
    max_chars=5000,
    help="Enter the message exactly as you received it. Do not include private information.",
)

if st.button("Analyze message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message before analyzing it.")
    else:
        try:
            message_features = vectorizer.transform([message.strip()])
            prediction = classifier.predict(message_features)[0]
            probabilities = classifier.predict_proba(message_features)[0]
            confidence = float(max(probabilities)) * 100
        except (ValueError, AttributeError) as error:
            st.error(f"The message could not be analyzed: {error}")
        else:
            if prediction == "spam":
                st.markdown(
                    '<div class="result spam"><strong>🚨 Spam Message</strong></div>',
                    unsafe_allow_html=True,
                )
                st.warning(
                    "This message looks like spam. Avoid clicking links, calling "
                    "unfamiliar numbers, or sharing personal or payment details."
                )
            else:
                st.markdown(
                    '<div class="result ham"><strong>✅ Not Spam Message</strong></div>',
                    unsafe_allow_html=True,
                )

            st.metric("Model confidence estimate", f"{confidence:.1f}%")
            st.caption(
                "This percentage is the model's estimated probability for its "
                "chosen label—not a guarantee or a calibrated measure of safety. "
                "A message classified as not spam can still be fraudulent."
            )

with st.expander("How does this analyzer work?"):
    st.write(
        "The project converts message text into TF-IDF features and uses a "
        "Multinomial Naive Bayes classifier trained on the UCI SMS Spam Collection. "
        "The model can make mistakes, so use your judgment with suspicious messages."
    )
