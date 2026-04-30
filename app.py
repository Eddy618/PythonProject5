import gradio as gr
import torch
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# ======================
# DOWNLOAD IMAGE MODEL
# ======================
def download_model():
    if not os.path.exists("image_model.h5"):
        url = "https://drive.google.com/uc?id=1G52iE_8FfaXx-6pyqxVeI3nAjoblDWKI"
        gdown.download(url, "image_model.h5", quiet=False)

download_model()

# ======================
# LOAD MODELS
# ======================
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
image_model = load_model("image_model.h5")

IMG_SIZE = 128

# ======================
# SCRAPE URL TEXT
# ======================
def extract_text_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text[:2000]  # limit size
    except:
        return "Unable to extract text from URL"

# ======================
# PREDICT FUNCTION
# ======================
def predict(text, url, image):

    # If URL is provided, override text
    if url:
        text = extract_text_from_url(url)

    if not text:
        return "No input provided", ""

    # -------- TEXT MODEL --------
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = text_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    text_pred = float(probs[0][1].item())

    # -------- IMAGE MODEL --------
    if image is not None:
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        image_pred = float(image_model.predict(image)[0][0])
    else:
        image_pred = 0.5  # neutral if no image

    # -------- FUSION --------
    final = (0.75 * text_pred) + (0.25 * image_pred)

    # -------- RESULT --------
    if final >= 0.75:
        result = "🟢 VERY LIKELY REAL NEWS"
    elif final >= 0.55:
        result = "🟡 POSSIBLY REAL NEWS"
    elif final >= 0.35:
        result = "🟠 POSSIBLY FAKE NEWS"
    else:
        result = "🔴 VERY LIKELY FAKE NEWS"

    return result, f"Text: {text_pred:.3f}\nImage: {image_pred:.3f}\nFinal: {final:.3f}"


# ======================
# GRADIO UI
# ======================
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="News Text"),
        gr.Textbox(label="News URL (optional)"),
        gr.Image(type="pil", label="Upload Image (optional)")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Scores")
    ],
    title="📰 Fake News Detector AI",
    description="Text + Image + Live URL Analysis (BERT + CNN)"
)

demo.launch(server_name="0.0.0.0", server_port=7860)