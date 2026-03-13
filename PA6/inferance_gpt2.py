
import torch
from transformers import pipeline, set_seed
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

import torch
import numpy as np

##//@@ Import opencl_functions

# original_forward = torch.nn.Linear.forward

def custom_forward(self, input=None, bias=None):
    output_shape = input.shape[:-1] + (self.weight.shape[::-1][-1],)
    input = input.reshape(-1, input.shape[-1])

    ##//@@ replace the below line with what is said in the docs
    output = torch.zeros(output_shape).to(input.device)  ## REPLACE THIS LINE

    output = output.reshape(output_shape)
    return output

def run_gpt_inference():
    og_forward = torch.nn.Linear.forward
    torch.nn.Linear.forward = custom_forward
    set_seed(42)

    generator = pipeline('text-generation', model='gpt2')
    out = generator("Hello, I'm a language model,", max_new_tokens=20)
    torch.nn.Linear.forward = og_forward
    return out[0]["generated_text"]

if __name__ == "__main__":
    print(run_gpt_inference())