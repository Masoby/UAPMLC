import streamlit as st
import numpy as np
import pickle
import torch
from PIL import Image
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Fake News Detection Dashboard",
    layout="wide"
)

MAX_LEN = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
label_map = {0: "FAKE", 1: "REAL"}

# =========================
# HEADER IMAGE (DI BAWAH TITLE)
# =========================
from PIL import Image
header_img = Image.open("assets/as.jpg")
st.image(header_img, use_container_width=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_lstm():
    model = load_model("models/lstm/lstm_fake_news_model.h5")
    with open("models/lstm/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


@st.cache_resource
def load_transformer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


lstm_model, lstm_tokenizer = load_lstm()
bert_model, bert_tokenizer = load_transformer("models/bert")
distilbert_model, distilbert_tokenizer = load_transformer("models/distilbert")

# =========================
# PREDICTION FUNCTIONS
# =========================
def predict_lstm(text):
    seq = lstm_tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    prob = lstm_model.predict(pad, verbose=0)[0][0]
    label = 1 if prob >= 0.5 else 0
    confidence = prob if label == 1 else 1 - prob
    return label, confidence


def predict_transformer(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, label = torch.max(probs, dim=1)

    return label.item(), confidence.item()


def ensemble_predict(text):
    preds = []
    confs = []

    for model, tok in [
        ("lstm", None),
        ("bert", None),
        ("distil", None)
    ]:
        if model == "lstm":
            l, c = predict_lstm(text)
        elif model == "bert":
            l, c = predict_transformer(bert_model, bert_tokenizer, text)
        else:
            l, c = predict_transformer(distilbert_model, distilbert_tokenizer, text)

        preds.append(l)
        confs.append(c)

    final_label = max(set(preds), key=preds.count)
    avg_conf = np.mean(confs)

    return final_label, avg_conf

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    ["LSTM", "BERT", "DistilBERT", "Ensemble (Majority Vote)"]
)

st.sidebar.header("ℹ️ Model Information")

st.sidebar.markdown(
    """
    **Label:**
    - 0 = Fake  
    - 1 = Real  

    ---

    **Model Accuracy (Test Set):**
    - **LSTM** : 81%
    - **BERT** : 96%
    - **DistilBERT** : 96%

    ---

    🏆 **Best Model:**  
    **BERT**  
    *(Highest accuracy and balanced precision–recall)*

    ---
    **Ensemble (Majority Vote):**
    Combines predictions from all models  
    for more robust decision making.
    """
)


# =========================
# UI
# =========================
st.title("📰 Fake News Detection Dashboard")

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch CSV Prediction"])

# =========================
# SINGLE TEXT PREDICTION
# =========================
with tab1:
    text_input = st.text_area(
        "✍️ Enter news text (English)",
        height=200
    )

    if st.button("Predict"):
        if text_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            if selected_model == "LSTM":
                label, conf = predict_lstm(text_input)
            elif selected_model == "BERT":
                label, conf = predict_transformer(
                    bert_model, bert_tokenizer, text_input
                )
            elif selected_model == "DistilBERT":
                label, conf = predict_transformer(
                    distilbert_model, distilbert_tokenizer, text_input
                )
            else:
                label, conf = ensemble_predict(text_input)

            st.success(f"Prediction: **{label_map[label]}**")
            st.info(f"Confidence: **{conf:.3f}**")

# =========================
# CSV BATCH PREDICTION
# =========================
with tab2:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview Data", df.head())

        text_column = st.selectbox(
            "Select text column",
            df.columns
        )

        if st.button("Run Batch Prediction"):
            predictions = []
            confidences = []

            for text in df[text_column].astype(str):
                if selected_model == "LSTM":
                    label, conf = predict_lstm(text)
                elif selected_model == "BERT":
                    label, conf = predict_transformer(
                        bert_model, bert_tokenizer, text
                    )
                elif selected_model == "DistilBERT":
                    label, conf = predict_transformer(
                        distilbert_model, distilbert_tokenizer, text
                    )
                else:
                    label, conf = ensemble_predict(text)

                predictions.append(label_map[label])
                confidences.append(conf)

            df["Prediction"] = predictions
            df["Confidence"] = confidences

            st.success("Batch prediction completed!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Result CSV",
                csv,
                "prediction_results.csv",
                "text/csv"
            )
