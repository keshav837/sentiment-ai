import streamlit as st
import joblib
import re

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SentimentAI",
    page_icon="🤖",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    max-width: 850px;
    padding-top: 3rem;
}

.hero {
    text-align: center;
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    opacity: 0.85;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.positive {
    background-color: #dcfce7;
    border: 1px solid #22c55e;
}

.negative {
    background-color: #fee2e2;
    border: 1px solid #ef4444;
}

.tech-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")


model = load_model()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# PREDICTION
# =========================================================

def predict_sentiment(text):

    cleaned_text = clean_text(text)

    prediction = model.predict(
        [cleaned_text]
    )[0]

    probabilities = model.predict_proba(
        [cleaned_text]
    )[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<h1>🤖 SentimentAI</h1>

<p>
Machine Learning Based Sentiment Analysis
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown(
"""
### 🧠 Analyze Your Text

Enter any **English sentence or review** below.
The machine learning model will classify it as:

- 😊 Positive
- 😞 Negative

and provide a confidence score.
"""
)


# =========================================================
# INPUT
# =========================================================

text = st.text_area(
    "Enter your text",
    height=150,
    placeholder="Example: I absolutely loved this product!"
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        with st.spinner(
            "Analyzing sentiment..."
        ):

            sentiment, confidence = predict_sentiment(
                text
            )

        # =============================================
        # POSITIVE
        # =============================================

        if sentiment == "positive":

            st.markdown(
                f"""
                <div class="result-card positive">

                <h2>😊 POSITIVE</h2>

                <p>
                The model detected a positive sentiment.
                </p>

                <h3>
                Confidence: {confidence:.2f}%
                </h3>

                </div>
                """,
                unsafe_allow_html=True
            )

        # =============================================
        # NEGATIVE
        # =============================================

        else:

            st.markdown(
                f"""
                <div class="result-card negative">

                <h2>😞 NEGATIVE</h2>

                <p>
                The model detected a negative sentiment.
                </p>

                <h3>
                Confidence: {confidence:.2f}%
                </h3>

                </div>
                """,
                unsafe_allow_html=True
            )

        # =============================================
        # CONFIDENCE BAR
        # =============================================

        st.write("### Confidence")

        st.progress(
            int(confidence)
        )


# =========================================================
# TECHNICAL INFORMATION
# =========================================================

st.markdown(
"""
<div class="tech-card">

<h3>⚙️ Model Information</h3>

<b>Algorithm:</b> Logistic Regression<br>

<b>Feature Extraction:</b> TF-IDF<br>

<b>Task:</b> Binary Sentiment Classification<br>

<b>Classes:</b> Positive / Negative<br>

<b>Dataset:</b> IMDb Movie Reviews<br>

<b>Language:</b> English

</div>
""",
unsafe_allow_html=True
)


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown("### 🔬 How It Works")

st.code(
"""
User Text
    ↓
Text Cleaning
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Positive / Negative
    ↓
Confidence Score
""",
language="text"
)


# =========================================================
# EXAMPLES
# =========================================================

st.markdown("### 🧪 Try These Examples")

col1, col2 = st.columns(2)

with col1:

    st.info(
        "😊 I absolutely loved this movie!"
    )

with col2:

    st.error(
        "😞 This movie was terrible."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
"""
<div class="footer">

<hr>

<p>
Built using Python • Scikit-learn • Streamlit
</p>

<p>
SentimentAI — Machine Learning Project
</p>

</div>
""",
unsafe_allow_html=True
  )
