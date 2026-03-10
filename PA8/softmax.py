
import torch
import numpy as np
from opencl_functions import ocl_softmax

def softmax(input, dim):
    print("and they don't stop comin'")
    input = torch.transpose(input, dim, -1)
    shape = input.shape
    input = input.reshape((np.prod(shape[:-1]), shape[-1]))

    # Students implement code below in OpenCL
    out = ocl_softmax(input)
    out = out.reshape(shape)

    return torch.transpose(out, dim, -1)
