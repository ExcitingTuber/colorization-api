from flask import Flask, request, render_template_string
import numpy as np
import cv2
import base64
from colorize import colorize_image

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
        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))

        output = colorize_image(img)

        _, buffer = cv2.imencode('.jpg', output)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return render_template_string(HTML_PAGE, output=img_base64)

    return render_template_string(HTML_PAGE)

if __name__ == "__main__":
    app.run()
