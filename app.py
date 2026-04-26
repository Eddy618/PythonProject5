import streamlit as st
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

def download_model():
    if not os.path.exists("image_model.h5"):
        url = "https://drive.google.com/uc?id=1G52iE_8FfaXx-6pyqxVeI3nAjoblDWKI"
        gdown.download(url, "image_model.h5", quiet=False)

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="📰",
    layout="centered"
)

# ======================
# NETFLIX STYLE UI
# ======================
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(120deg, #0f0f0f, #1a1a1a);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #00ffd5;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #bbbbbb;
    margin-bottom: 30px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,255,213,0.1);
    margin-bottom: 20px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00ffd5, #008cff);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px;
    width: 100%;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 15px #00ffd5;
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown("<div class='title'>📰 Fake News Detector AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>BERT + Image AI Multi-Modal Detection System</div>", unsafe_allow_html=True)

# ======================
# LOAD MODELS
# ======================
@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizer.from_pretrained("bert_model")
    model = DistilBertForSequenceClassification.from_pretrained("bert_model")

    @st.cache_resource
    def load_models():
        download_model()  # 👈 ADD THIS LINE

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        image_model = load_model("image_model.h5")

        return tokenizer, model, image_model
    return tokenizer, model, image_model

st.write("Loading models...")
tokenizer, model, image_model = load_models()
st.write("Models loaded successfully!")

IMG_SIZE = 128

# ======================
# PREDICT FUNCTION
# ======================

def predict(text, image):

    # -------- TEXT (BERT) --------
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    text_pred = float(probs[0][1].item())


    # -------- IMAGE --------
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    image_pred = float(image_model.predict(image)[0][0])

    # -------- FUSION (IMPROVED) --------
    final = (0.75 * text_pred) + (0.25 * image_pred)

    # -------- RESULT LOGIC --------
    if final >= 0.75:
        result = "🟢 VERY LIKELY REAL NEWS"
    elif final >= 0.55:
        result = "🟡 POSSIBLY REAL NEWS"
    elif final >= 0.35:
        result = "🟠 POSSIBLY FAKE NEWS"
    else:
        result = "🔴 VERY LIKELY FAKE NEWS"

    return result, final, text_pred, image_pred

def generate_explanation(text, text_score, image_score, final_score):

    reasons = []

    # TEXT ANALYSIS
    if text_score < 0.4:
        reasons.append("Text content appears unreliable or misleading")
    elif text_score < 0.6:
        reasons.append("Text contains uncertain or exaggerated language")
    else:
        reasons.append("Text appears credible")

    # IMAGE ANALYSIS
    if image_score < 0.4:
        reasons.append("Image analysis confidence is low")
    elif image_score < 0.6:
        reasons.append("Image is somewhat unclear or not strongly supportive")
    else:
        reasons.append("Image supports the content")

    # FINAL CONFIDENCE
    if final_score < 0.4:
        reasons.append("Overall model confidence is low — likely fake")
    elif final_score < 0.6:
        reasons.append("Model is uncertain — verify with trusted sources")
    else:
        reasons.append("Model has high confidence in authenticity")

    return reasons
# ======================
# INPUT UI (CARDS)
# ======================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### ✍️ News Text")
text_input = st.text_area("", placeholder="Type or paste news here...")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🖼 Upload Image")
image_file = st.file_uploader("", type=["jpg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# ======================
# BUTTON + OUTPUT
# ======================
if st.button("🔍 Analyze News"):

    if text_input and image_file:

        image = Image.open(image_file)

        result, score, text_score, image_score = predict(text_input, image)

        explanations = generate_explanation(
            text_input,
            text_score,
            image_score,
            score
        )

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown("### 🧠 Prediction Result")

        if "FAKE" in result:
            st.error(result)
        else:
            st.success(result)

        st.markdown(f"### 📊 Confidence Score: `{round(score, 3)}`")
        st.progress(float(score))
        st.markdown("### 🧠 Why this result?")

        for reason in explanations:
            st.write(f"- {reason}")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Please enter text and upload an image.")


tokenizer, model, image_model = load_models()
