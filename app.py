import base64
import io

import numpy as np
import onnxruntime as ort

from PIL import Image

from flask import Flask, render_template, request


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD ONNX MODEL
# ==========================================

session = ort.InferenceSession(
    "handwriting_model.onnx"
)


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

    # --------------------------------------
    # RECEIVE IMAGE
    # --------------------------------------

    data = request.get_json()

    if not data or "image" not in data:

        return {
            "error": "No image received"
        }, 400


    image_data = data["image"]


    # --------------------------------------
    # DECODE BASE64 IMAGE
    # --------------------------------------

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


    # --------------------------------------
    # ADD WHITE BACKGROUND
    # --------------------------------------

    background = Image.new(
        "RGBA",
        image.size,
        "white"
    )

    background.alpha_composite(
        image.convert("RGBA")
    )


    # --------------------------------------
    # GRAYSCALE
    # --------------------------------------

    grayscale_image = background.convert(
        "L"
    )


    # --------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------

    image_array = np.array(
        grayscale_image
    )


    # --------------------------------------
    # FIND HANDWRITING
    # --------------------------------------

    ink_pixels = image_array < 250

    rows, columns = np.where(
        ink_pixels
    )


    if len(rows) == 0:

        return {
            "prediction": "Draw a digit first"
        }


    # --------------------------------------
    # FIND BOUNDING BOX
    # --------------------------------------

    top = rows.min()
    bottom = rows.max()

    left = columns.min()
    right = columns.max()


    # --------------------------------------
    # CROP
    # --------------------------------------

    cropped_image = grayscale_image.crop(
        (
            left,
            top,
            right + 1,
            bottom + 1
        )
    )


    # --------------------------------------
    # PRESERVE ASPECT RATIO
    # --------------------------------------

    width, height = cropped_image.size

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


    # --------------------------------------
    # CREATE 28 x 28 IMAGE
    # --------------------------------------

    final_image = Image.new(
        "L",
        (28, 28),
        255
    )


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


    # --------------------------------------
    # NORMALISE
    # --------------------------------------

    final_array = np.array(
        final_image
    ).astype(
        np.float32
    )


    final_array = (
        final_array /
        255.0
    )


    # MNIST has white digits on
    # a black background
    final_array = (
        1.0 -
        final_array
    )


    # --------------------------------------
    # CREATE CNN INPUT
    # --------------------------------------

    # Current shape:
    #
    # [28, 28]
    #
    # ONNX expects:
    #
    # [1, 1, 28, 28]

    final_array = np.expand_dims(
        final_array,
        axis=0
    )


    final_array = np.expand_dims(
        final_array,
        axis=0
    )


    # --------------------------------------
    # ONNX PREDICTION
    # --------------------------------------

    outputs = session.run(
        None,
        {
            "input": final_array
        }
    )


    # outputs[0] contains the CNN logits
    logits = outputs[0]


    prediction = int(
        np.argmax(
            logits,
            axis=1
        )[0]
    )


    # --------------------------------------
    # CALCULATE CONFIDENCE
    # --------------------------------------

    # Convert logits to probabilities
    # using softmax.

    shifted_logits = (
        logits -
        np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )


    exponentials = np.exp(
        shifted_logits
    )


    probabilities = (
        exponentials /
        np.sum(
            exponentials,
            axis=1,
            keepdims=True
        )
    )


    confidence = float(
        probabilities[
            0,
            prediction
        ]
    )


    confidence = round(
        confidence * 100,
        1
    )


    # --------------------------------------
    # RETURN RESULT
    # --------------------------------------

    return {
        "prediction": prediction,
        "confidence": confidence
    }


# ==========================================
# RUN LOCALLY
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)