import gradio as gr
import numpy as np
import cv2
from colorize import colorize_image

def process_image(input_image):
    """Takes an RGB image from Gradio, colorizes it, returns RGB output."""
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    # Gradio gives us RGB numpy array, OpenCV needs BGR
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Run colorization
    output_bgr = colorize_image(img_bgr)

    # Convert back to RGB for Gradio
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    return output_rgb

# Build the Gradio interface
with gr.Blocks(
    title="Image Colorization",
    theme=gr.themes.Soft(primary_hue="violet"),
    css="""
        .gradio-container { max-width: 900px !important; margin: auto; }
        h1 { text-align: center; margin-bottom: 0.5em; }
        p.subtitle { text-align: center; color: #666; margin-top: 0; }
    """
) as demo:
    gr.Markdown("# 🎨 Image Colorization")
    gr.Markdown(
        "<p class='subtitle'>Upload a black & white image and watch it come to life with AI-powered colorization</p>"
    )

    with gr.Row():
        input_img = gr.Image(label="Upload B&W Image", type="numpy")
        output_img = gr.Image(label="Colorized Result", type="numpy")

    colorize_btn = gr.Button("✨ Colorize", variant="primary", size="lg")
    colorize_btn.click(fn=process_image, inputs=input_img, outputs=output_img)

    gr.Examples(
        examples=[],  # Add example image paths here if you want
        inputs=input_img,
    )

    gr.Markdown(
        "---\n"
        "**How it works:** This uses a deep learning model (Zhang et al.) trained on ImageNet "
        "to predict color channels from grayscale input."
    )

if __name__ == "__main__":
    demo.launch()
