import gradio as gr
import numpy as np
import cv2
import tempfile
from colorize import colorize_image
from upscale import upscale_image


def save_image_for_download(img_bgr, prefix="result"):
    """Save a BGR image to a temp PNG file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix=f"{prefix}_", delete=False)
    cv2.imwrite(tmp.name, img_bgr)
    tmp.close()
    return tmp.name


def process_colorize(input_image):
    """Takes an RGB image from Gradio, colorizes it, returns RGB output."""
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output_bgr = colorize_image(img_bgr)
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    h, w = output_rgb.shape[:2]
    download_path = save_image_for_download(output_bgr, "colorized")
    return output_rgb, f"Output: {w} × {h} px", download_path


def process_upscale(input_image, scale, enhance_faces):
    """Upscale an image using Real-ESRGAN."""
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    scale_int = int(scale.replace("x", ""))
    output_bgr = upscale_image(img_bgr, scale=scale_int, enhance_faces=enhance_faces)
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    h_in, w_in = input_image.shape[:2]
    h_out, w_out = output_rgb.shape[:2]
    info = f"{w_in}×{h_in} → {w_out}×{h_out} (Real-ESRGAN {scale})"
    if enhance_faces:
        info += " + GFPGAN face restore"
    download_path = save_image_for_download(output_bgr, "hd_upscale")
    return output_rgb, info, download_path


def process_colorize_and_upscale(input_image, scale, enhance_faces):
    """Colorize a B&W image then upscale with Real-ESRGAN."""
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    colorized_bgr = colorize_image(img_bgr)
    scale_int = int(scale.replace("x", ""))
    output_bgr = upscale_image(colorized_bgr, scale=scale_int, enhance_faces=enhance_faces)
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    h_in, w_in = input_image.shape[:2]
    h_out, w_out = output_rgb.shape[:2]
    info = f"Colorized + Enhanced: {w_in}×{h_in} → {w_out}×{h_out}"
    download_path = save_image_for_download(output_bgr, "colorized_hd")
    return output_rgb, info, download_path


# Build the Gradio interface
with gr.Blocks(
    title="Image Colorization & HD Upscale",
    theme=gr.themes.Soft(primary_hue="violet"),
    css="""
        .gradio-container { max-width: 960px !important; margin: auto; }
        h1 { text-align: center; margin-bottom: 0.2em; }
        p.subtitle { text-align: center; color: #666; margin-top: 0; }
        .res-info { font-family: monospace; font-size: 0.9em; }
    """
) as demo:
    gr.Markdown("# 🎨 Image Colorization & HD Restore")
    gr.Markdown(
        "<p class='subtitle'>Colorize B&W photos and restore image quality with "
        "Real-ESRGAN super-resolution + GFPGAN face enhancement</p>"
    )

    with gr.Tabs():
        # --- Tab 1: Colorize ---
        with gr.TabItem("🎨 Colorize"):
            with gr.Row():
                c_input = gr.Image(label="Upload B&W Image", type="numpy")
                c_output = gr.Image(label="Colorized Result", type="numpy")
            c_info = gr.Textbox(label="Resolution", interactive=False, elem_classes="res-info")
            c_btn = gr.Button("✨ Colorize", variant="primary", size="lg")
            c_download = gr.File(label="📥 Download Colorized Image")
            c_btn.click(fn=process_colorize, inputs=c_input, outputs=[c_output, c_info, c_download])

        # --- Tab 2: HD Restore ---
        with gr.TabItem("🔍 HD Restore"):
            with gr.Row():
                u_input = gr.Image(label="Upload Image", type="numpy")
                u_output = gr.Image(label="Restored HD Result", type="numpy")
            with gr.Row():
                u_scale = gr.Radio(["2x", "4x"], value="4x", label="Upscale Factor")
                u_faces = gr.Checkbox(label="🧑 Enhance Faces (GFPGAN)", value=False)
            u_info = gr.Textbox(label="Resolution", interactive=False, elem_classes="res-info")
            u_btn = gr.Button("🔍 Restore & Upscale", variant="primary", size="lg")
            u_download = gr.File(label="📥 Download HD Image")
            u_btn.click(
                fn=process_upscale,
                inputs=[u_input, u_scale, u_faces],
                outputs=[u_output, u_info, u_download],
            )

        # --- Tab 3: Colorize + HD ---
        with gr.TabItem("🚀 Colorize + HD"):
            gr.Markdown("*Upload a B&W image to colorize **and** restore in one step.*")
            with gr.Row():
                cu_input = gr.Image(label="Upload B&W Image", type="numpy")
                cu_output = gr.Image(label="Colorized HD Result", type="numpy")
            with gr.Row():
                cu_scale = gr.Radio(["2x", "4x"], value="4x", label="Upscale Factor")
                cu_faces = gr.Checkbox(label="🧑 Enhance Faces (GFPGAN)", value=False)
            cu_info = gr.Textbox(label="Resolution", interactive=False, elem_classes="res-info")
            cu_btn = gr.Button("🚀 Colorize + Restore", variant="primary", size="lg")
            cu_download = gr.File(label="📥 Download Colorized HD Image")
            cu_btn.click(
                fn=process_colorize_and_upscale,
                inputs=[cu_input, cu_scale, cu_faces],
                outputs=[cu_output, cu_info, cu_download],
            )

    gr.Markdown(
        "---\n"
        "**Models:** Colorization — Zhang et al. (2016) · "
        "Super-Resolution — [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (x4plus) · "
        "Face Restore — [GFPGAN](https://github.com/TencentARC/GFPGAN) v1.4"
    )

if __name__ == "__main__":
    demo.launch()
