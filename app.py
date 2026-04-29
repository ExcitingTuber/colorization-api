from flask import Flask, request, Response
import numpy as np
import cv2
from colorize import colorize_image

app = Flask(__name__)

@app.route("/")
def home():
    return "Cloud Colorization API is running!"

@app.route("/colorize", methods=["POST"])
def colorize():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    output = colorize_image(img)

    _, buffer = cv2.imencode('.jpg', output)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)