from flask import Flask, request, render_template_string
import joblib
import re

app = Flask(__name__)

model = joblib.load("sentiment_model.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SentimentAI</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1">

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: white;
        }

        .container {
            max-width: 800px;
            margin: auto;
            padding: 30px 20px;
        }

        .hero {
            text-align: center;
            padding: 35px 20px;
            background: linear-gradient(
                135deg,
                #1e3a8a,
                #312e81
            );
            border-radius: 20px;
        }

        .hero h1 {
            font-size: 42px;
            margin: 0;
        }

        .hero p {
            color: #cbd5e1;
        }

        .card {
            margin-top: 25px;
            background: #1e293b;
            padding: 25px;
            border-radius: 18px;
        }

        textarea {
            width: 100%;
            height: 150px;
            padding: 15px;
            border-radius: 12px;
            border: none;
            font-size: 16px;
            resize: vertical;
        }

        button {
            width: 100%;
            margin-top: 15px;
            padding: 15px;
            border: none;
            border-radius: 12px;
            background: #2563eb;
            color: white;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .positive {
            margin-top: 20px;
            padding: 20px;
            text-align: center;
            border-radius: 15px;
            background: #14532d;
        }

        .negative {
            margin-top: 20px;
            padding: 20px;
            text-align: center;
            border-radius: 15px;
            background: #7f1d1d;
        }

        .info {
            line-height: 1.8;
            color: #cbd5e1;
        }

        footer {
            text-align: center;
            margin-top: 30px;
            color: #94a3b8;
        }

    </style>
</head>

<body>

<div class="container">

    <div class="hero">

        <h1>🤖 SentimentAI</h1>

        <p>
            Machine Learning Based Sentiment Analysis
        </p>

    </div>


    <div class="card">

        <h2>Analyze Your Text</h2>

        <form method="POST">

            <textarea
                name="text"
                placeholder="Example: I really loved this product!"
                required
            ></textarea>

            <button type="submit">
                🔍 Analyze Sentiment
            </button>

        </form>


        {% if result %}

            {% if result == "positive" %}

                <div class="positive">

                    <h2>😊 POSITIVE</h2>

                    <p>
                        Confidence: {{ confidence }}%
                    </p>

                </div>

            {% else %}

                <div class="negative">

                    <h2>😞 NEGATIVE</h2>

                    <p>
                        Confidence: {{ confidence }}%
                    </p>

                </div>

            {% endif %}

        {% endif %}

    </div>


    <div class="card">

        <h2>⚙️ Model Information</h2>

        <div class="info">

            <b>Algorithm:</b>
            Logistic Regression

            <br>

            <b>Feature Extraction:</b>
            TF-IDF

            <br>

            <b>Task:</b>
            Binary Sentiment Classification

            <br>

            <b>Dataset:</b>
            IMDb Movie Reviews

            <br>

            <b>Classes:</b>
            Positive / Negative

        </div>

    </div>


    <footer>

        Built with Python + Flask + Scikit-learn

    </footer>

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None

    if request.method == "POST":

        text = request.form["text"]

        cleaned = clean_text(text)

        result = model.predict([cleaned])[0]

        probabilities = model.predict_proba([cleaned])[0]

        confidence = round(
            max(probabilities) * 100,
            2
        )

    return render_template_string(
        HTML,
        result=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run()
