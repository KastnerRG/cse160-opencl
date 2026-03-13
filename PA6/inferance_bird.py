""" Run inference on a hastly made Peru Bird Classifier.

For seeing how this model was trained see extras/train_audio.py
"""
from transformers import AutoModel
import torch
from torch.nn import functional as F
import datasets
import torch.nn as nn
from opencl_functions import ocl_conv2d
import pytorch_ocl
from extras.train_audio import preprocess_function, PeruBirdClassifier
from utils.select_device import select_device
from safetensors.torch import load_file
from utils.conv2d_forward import my__conv_forward
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

def run_bird_inference(device=None, swap_conv_forward=True):
    og__conv_forward = torch.nn.Conv2d._conv_forward
    if swap_conv_forward:
        torch.nn.Conv2d._conv_forward = my__conv_forward

    if device is None:
        device = select_device()

    selected_birds = datasets.load_from_disk("Dataset/audio/selected_bird_datasets")    
    selected_birds.set_transform(preprocess_function)

    checkpoint_path = "Dataset/audio/model/OCLBirdClassifer/model.safetensors"
    state_dict = load_file(checkpoint_path)
    model = PeruBirdClassifier(num_classes=132, device="cpu").to(device)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        input_ids, labels = selected_birds[:]["input_ids"], selected_birds[:]["labels"]
        output = model(torch.tensor(input_ids).to(device), labels=torch.tensor(labels).to(device))
        preds = torch.argmax(output["logits"], dim=-1)
    
    torch.nn.Conv2d._conv_forward = og__conv_forward

    return list(preds.cpu().numpy().tolist())
    

if __name__ == "__main__":
    print(run_bird_inference())
