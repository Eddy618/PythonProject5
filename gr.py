import gradio as gr
import torch

# example fake model function
def predict(text):
    return "Model output for: " + text

app = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text"
)

app.launch()
