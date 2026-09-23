import base64
import io

import numpy as np
import torch

from PIL import Image

from flask import Flask, render_template, request

from model import HandwritingCNN


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD MODEL
# ==========================================

model = HandwritingCNN()


model.load_state_dict(
    torch.load(
        "handwriting_model.pth",
        map_location="cpu"
    )
)


model.eval()


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICTION
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Get JSON from JavaScript
    data = request.get_json()


    if not data or "image" not in data:

        return {
            "error": "No image received"
        }, 400


    image_data = data["image"]


    # ======================================
    # DECODE IMAGE
    # ======================================

    try:

        header, encoded = image_data.split(
            ",",
            1
        )

        image_bytes = base64.b64decode(
            encoded
        )

        image = Image.open(
            io.BytesIO(image_bytes)
        )

    except Exception:

        return {
            "error": "Invalid image"
        }, 400


    # ======================================
    # WHITE BACKGROUND
    # ======================================

    background = Image.new(
        "RGBA",
        image.size,
        "white"
    )


    background.alpha_composite(
        image.convert("RGBA")
    )


    # ======================================
    # GRAYSCALE
    # ======================================

    grayscale_image = background.convert(
        "L"
    )


    # ======================================
    # NUMPY ARRAY
    # ======================================

    image_array = np.array(
        grayscale_image
    )


    # ======================================
    # FIND HANDWRITING
    # ======================================

    ink_pixels = image_array < 250


    rows, columns = np.where(
        ink_pixels
    )


    # Blank canvas
    if len(rows) == 0:

        return {
            "prediction": "Draw a digit first"
        }


    # ======================================
    # BOUNDING BOX
    # ======================================

    top = rows.min()
    bottom = rows.max()

    left = columns.min()
    right = columns.max()


    # ======================================
    # CROP
    # ======================================

    cropped_image = grayscale_image.crop(
        (
            left,
            top,
            right + 1,
            bottom + 1
        )
    )


    # ======================================
    # PRESERVE ASPECT RATIO
    # ======================================

    width, height = cropped_image.size


    # MNIST digits fit inside roughly
    # a 20 x 20 region of a 28 x 28 image.

    max_size = 20


    if width > height:

        new_width = max_size

        new_height = max(
            1,
            round(
                height *
                max_size /
                width
            )
        )

    else:

        new_height = max_size

        new_width = max(
            1,
            round(
                width *
                max_size /
                height
            )
        )


    resized_digit = cropped_image.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )


    # ======================================
    # CREATE 28 x 28 WHITE IMAGE
    # ======================================

    final_image = Image.new(
        "L",
        (28, 28),
        255
    )


    # Centre digit
    x_position = (
        28 - new_width
    ) // 2

    y_position = (
        28 - new_height
    ) // 2


    final_image.paste(
        resized_digit,
        (
            x_position,
            y_position
        )
    )


    # ======================================
    # CONVERT TO NUMPY
    # ======================================

    final_array = np.array(
        final_image
    ).astype(
        np.float32
    )


    # ======================================
    # NORMALISE
    # ======================================

    final_array = (
        final_array /
        255.0
    )


    # ======================================
    # INVERT
    # ======================================

    # White background -> 0
    # Dark handwriting -> 1

    final_array = (
        1.0 -
        final_array
    )


    # ======================================
    # PYTORCH TENSOR
    # ======================================

    tensor = torch.tensor(
        final_array,
        dtype=torch.float32
    )


    # [28, 28]
    # ->
    # [1, 28, 28]

    tensor = tensor.unsqueeze(0)


    # [1, 28, 28]
    # ->
    # [1, 1, 28, 28]

    tensor = tensor.unsqueeze(0)


    # ======================================
    # PREDICTION
    # ======================================

    with torch.no_grad():

        outputs = model(
            tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()


        confidence = probabilities[
            0,
            prediction
        ].item()


    # Convert confidence to percentage
    confidence = round(
        confidence * 100,
        1
    )


    # ======================================
    # SEND RESULT TO JAVASCRIPT
    # ======================================

    return {
        "prediction": prediction,
        "confidence": confidence
    }


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)