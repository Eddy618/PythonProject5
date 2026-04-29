print("APP STARTED")
import streamlit as st
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="📰",
    layout="centered"
)

# ======================
# STYLE
# ======================
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    color: #00ffd5;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown("<div class='title'>📰 Fake News Detector AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>BERT AI News Verification</div>", unsafe_allow_html=True)

# ======================
# LOAD MODEL (TEXT ONLY)
# ======================
@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

st.write("Loading AI model...")
tokenizer, model = load_models()
st.success("Model loaded!")

# ======================
# PREDICT FUNCTION
# ======================
def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    score = float(probs[0][1].item())

    if score >= 0.75:
        result = "🟢 VERY LIKELY REAL NEWS"
    elif score >= 0.55:
        result = "🟡 POSSIBLY REAL NEWS"
    elif score >= 0.35:
        result = "🟠 POSSIBLY FAKE NEWS"
    else:
        result = "🔴 VERY LIKELY FAKE NEWS"

    return result, score

# ======================
# UI INPUT
# ======================
text_input = st.text_area("✍️ Enter News Text")

# ======================
# BUTTON
# ======================
if st.button("🔍 Analyze News"):

    if text_input:

        result, score = predict(text_input)

        st.subheader(result)
        st.write("Confidence Score:", round(score, 3))
        st.progress(score)

    else:
        st.warning("Please enter news text.")