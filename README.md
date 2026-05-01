🧠 Multi-Modal Fake News Detection App

A powerful AI-powered web application that detects fake news using both text and image analysis. This project combines Natural Language Processing (NLP) and Computer Vision to provide more accurate and robust predictions.

🚀 Features
📰 Text Analysis — Detect fake or real news from written content
🖼️ Image Analysis — Analyze images associated with news
🔗 URL Input (Optional) — Extendable to fetch and analyze news articles directly
⚡ Fast Web Interface — Built with Gradio for smooth user interaction
🌍 Deployed Online — Accessible via Railway
🛠️ Tech Stack
Frontend/UI: Gradio
Backend: Python
NLP Model: DistilBERT (distilbert-base-uncased)
Image Model: TensorFlow / Keras (.h5 model)
Deployment: Railway
Libraries:
transformers
torch
tensorflow
gradio
scikit-learn
📂 Project Structure
├── app.py                # Main application (Gradio UI)
├── requirements.txt     # Dependencies
├── Procfile             # Railway start command
├── image_model.h5       # Pretrained image model
├── utils/               # Helper functions (if any)
└── README.md            # Project documentation
