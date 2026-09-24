import numpy as np
import onnxruntime as ort


# Load the ONNX model
session = ort.InferenceSession(
    "handwriting_model.onnx"
)


# Create fake input with the same shape
# as a handwriting image
test_input = np.random.rand(
    1,
    1,
    28,
    28
).astype(np.float32)


# Run the neural network
outputs = session.run(
    None,
    {
        "input": test_input
    }
)


print(
    "Output shape:",
    outputs[0].shape
)


print(
    "ONNX model works!"
)