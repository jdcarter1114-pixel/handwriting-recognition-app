import torch

from model import HandwritingCNN


# ==========================================
# LOAD OUR TRAINED PYTORCH MODEL
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
# CREATE AN EXAMPLE INPUT
# ==========================================

# Our CNN expects:
#
# [batch, channel, height, width]
#
# [1, 1, 28, 28]

example_input = torch.randn(
    1,
    1,
    28,
    28
)


# ==========================================
# EXPORT TO ONNX
# ==========================================

torch.onnx.export(
    model,
    example_input,
    "handwriting_model.onnx",

    input_names=[
        "input"
    ],

    output_names=[
        "output"
    ],

    opset_version=18
)


print(
    "Model successfully converted "
    "to handwriting_model.onnx"
)