
import torch
from transformers import pipeline, set_seed
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

import torch
import numpy as np
from softmax import softmax

##//@@ Import opencl_functions
# from opencl_functions import ocl_matmul

# # original_forward = torch.nn.Linear.forward

# def custom_forward(self, input=None, bias=None):
#     output_shape = input.shape[:-1] + (self.weight.shape[::-1][-1],)
#     input = input.reshape(-1, input.shape[-1])

#     ##//@@ replace the below line with what is said in the docs
#     # output = torch.zeros(output_shape).to(input.device)   
#     output = ocl_matmul(input, self.weight.t())

#     output = output.reshape(output_shape)
#     return output

def run_gpt_inference(text = "Hello, I'm a language model,"):
    of_softmax = torch.nn.functional.softmax
    torch.nn.functional.softmax = softmax
    set_seed(42)

    generator = pipeline('text-generation', model='gpt2')
    out = generator(text, max_new_tokens=20)
    torch.nn.functional.softmax = of_softmax
    return out[0]["generated_text"]


def run_tinyllama_inference(text="Hello, I'm a language model,"):
    of_softmax = torch.nn.functional.softmax
    torch.nn.functional.softmax = softmax
    set_seed(42)
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="cpu",
        torch_dtype=torch.bfloat16
    )
    out = pipe(text, max_new_tokens=20)
    torch.nn.functional.softmax = of_softmax
    return out[0]["generated_text"]

def run_qwen_inference(text="Hello, I'm a language model,", max_tokens=20):
    of_softmax = torch.nn.functional.softmax
    torch.nn.functional.softmax = softmax
    set_seed(42)
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3.5-0.8B",
        device="cpu",
        torch_dtype=torch.bfloat16
    )
    out = pipe(text, max_new_tokens=max_tokens)
    torch.nn.functional.softmax = of_softmax
    return out[0]["generated_text"]

class CustomSoftmax(torch.nn.Softmax):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return softmax(input, self.dim)

if __name__ == "__main__":
    old_softmax = torch.nn.functional.softmax
    torch.nn.functional.softmax = softmax
    x = torch.randn((1, 10))
    y = torch.nn.functional.softmax(x, dim=1)
    print(y)
    torch.nn.functional.softmax = old_softmax
    print(torch.nn.functional.softmax(x, dim=1))
    
    
    # print(run_gpt_inference())
    # print(run_gpt_inference("I'm GPT2, Hi Tom!"))
    # print(run_tinyllama_inference("Who are you?"))
    # print(run_qwen_inference("Continue the song: Never gonna give you up, ", max_tokens=100))
    # print(run_qwen_inference("Come up with line of python code to implement the softmax equation", max_tokens=500))

    # m = torch.nn.Softmax(dim=1)
    # n = CustomSoftmax(dim=1)

    # print(m(x))
    # print("testing")
    # print(n(x))
