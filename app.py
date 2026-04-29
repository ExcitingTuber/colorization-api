from flask import Flask, request, render_template_string
import numpy as np
import cv2
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Colorization</title>
    <style>
        body {
            font-family: Arial;
            text-align: center;
            margin-top: 40px;
        }
        img {
            max-width: 300px;
            margin: 10px;
            border: 2px solid #ddd;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            margin-top: 10px;
            cursor: pointer;
        }
    </style>
</head>
<body>

<h2>Upload Black & White Image</h2>

<form method="POST" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required onchange="previewImage(event)">
    <br><br>

    <img id="preview" style="display:none;"/>

    <br>
    <button type="submit" id="colorBtn" style="display:none;">Colorize</button>
</form>

{% if error %}
    <h3 style="color: red;">Error: {{ error }}</h3>
{% endif %}

{% if output %}
    <h3>Colorized Image</h3>
    <img src="data:image/jpeg;base64,{{output}}" />
{% endif %}

<script>
function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function(){
        const img = document.getElementById('preview');
        img.src = reader.result;
        img.style.display = 'block';

        document.getElementById('colorBtn').style.display = 'inline-block';
    }
    reader.readAsDataURL(event.target.files[0]);
}
</script>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            file = request.files['image']
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                return render_template_string(HTML_PAGE, error="Could not decode image. Please upload a valid image file.")

            # Import here to avoid import-time model loading issues
            from colorize import colorize_image
            output = colorize_image(img)

            _, buffer = cv2.imencode('.jpg', output)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template_string(HTML_PAGE, output=img_base64)
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return render_template_string(HTML_PAGE, error=str(e))

    return render_template_string(HTML_PAGE)

@app.route("/health")
def health():
    """Health check endpoint for Render."""
    return {"status": "ok"}, 200

if __name__ == "__main__":
    app.run()
