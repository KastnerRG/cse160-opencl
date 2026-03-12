""" Primarily from https://huggingface.co/docs/transformers/main/model_doc/resnet#transformers.ResNetForImageClassification.forward.example
"""

from transformers import AutoImageProcessor, ResNetForImageClassification, set_seed
import torch
from torch.nn import functional as F
from datasets import load_dataset
import torch.nn as nn
from opencl_functions import ocl_conv2d
import pytorch_ocl
from utils.conv2d_forward import my__conv_forward
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

def run_resnet_inference(device="cpu", image_path="", swap_conv_forward=True):
    og__conv_forward = torch.nn.Conv2d._conv_forward
    if swap_conv_forward:
        torch.nn.Conv2d._conv_forward = my__conv_forward

    set_seed(42)

    if image_path == "":
        dataset = load_dataset("huggingface/cats-image")
        image = dataset["test"]["image"][0]

    else:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device=device)

    inputs = image_processor(image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].to(device=device)

    # print(inputs['pixel_values'].device, model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    # print(model.config.id2label[predicted_label])
    # Correct output tiger cat

    #fix conv forward if reusing this
    torch.nn.Conv2d._conv_forward = og__conv_forward
    return model.config.id2label[predicted_label]

if __name__ == "__main__":
    run_inference()