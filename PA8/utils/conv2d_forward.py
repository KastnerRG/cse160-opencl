from transformers import AutoImageProcessor, ResNetForImageClassification, set_seed
import torch
from torch.nn import functional as F
from datasets import load_dataset
import torch.nn as nn
from opencl_functions import ocl_conv2d
import pytorch_ocl
# Going to replace the  conv_forward function in torch.nn.Conv2d
# https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/nn/modules/conv.py#L534C5-L550C10
def my__conv_forward(self, input, weight, bias):
    padding = self.padding
    if self.padding_mode != "zeros":
        input = F.pad(
            input, self._reversed_padding_repeated_twice, mode=self.padding_mode
        )
        padding =  _pair(0)
    
    # return F.conv2d(
    #     input, weight, bias, self.stride, padding, self.dilation, self.groups
    # )
    else:
        input = F.pad(input, pad=(padding[1], padding[1], padding[0], padding[0]), mode='constant', value=0)

    if bias is None:
        bias = torch.zeros(weight.shape[0])

    output = ocl_conv2d(
        input,
        weight,
        bias,
        self.stride[0],
        self.stride[1],
        self.dilation[0],
        self.dilation[1],
        self.groups
    )
    return output