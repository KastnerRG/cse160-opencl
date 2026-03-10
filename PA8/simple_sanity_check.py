from utils.select_device import select_device
import torch
import numpy as np
from softmax import softmax

if __name__ == "__main__":
    old_softmax = torch.nn.functional.softmax
    torch.nn.functional.softmax = softmax
    x = torch.randn((1, 10))
    y = torch.nn.functional.softmax(x, dim=1)
    print(y)
    torch.nn.functional.softmax = old_softmax
    print(torch.nn.functional.softmax(x, dim=1))
