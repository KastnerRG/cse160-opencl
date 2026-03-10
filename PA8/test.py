import time
import json
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import torch.nn as nn
# from opencl_functions import ocl_conv2d
import pytorch_ocl
import numpy as np
import os
from transformers import set_seed

from utils.select_device import select_device
# from Dataset.image.images import get_legally_distant_character, get_pet_photos
# from inferance_resnet import run_resnet_inference
# from inferance_bird import run_bird_inference
from inferance_gpt2 import run_gpt_inference, run_tinyllama_inference, run_qwen_inference
from softmax import softmax

class CustomSoftmax(torch.nn.Softmax):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return softmax(input, self.dim)

class TestSoftmax:
    print("well the years start comin'")
    """A class to test the correctness of the Softmax implementation."""
    def __init__(self, device):
        self.device = device


    def test(self, softmax_ref, softmax_test, input_tensor):
        result = {}

        output_reference = softmax_ref(input_tensor)
        output_test = softmax_test(input_tensor)

        result["shape"] = {
            "pass": "PASS" if output_reference.shape == output_test.shape else "FAIL",
            "reference": output_reference.shape,
            "student_implementation": output_test.shape  
        }
        
        result["value"] = {                     # Pytorch_OCL doesn't support allclose :(
            "pass": "PASS" if torch.allclose(output_reference.cpu(), output_test.cpu(), rtol=1e-5, atol=1e-5) else "FAIL"
        }
       
        return result

    def test_return_output(self, softmax_ref, softmax_test, input_tensor):
        output_reference = softmax_ref(input_tensor)
        output_test = softmax_test(input_tensor)

        
        return output_reference, output_test
       

    def test_softmax(self, input_tensor, dim=-1):
        """Test the forward pass of the OclConv2d layer using randomly initalized weights."""
        softmax_ref = nn.Softmax(dim).to(self.device)
        softmax_test = CustomSoftmax(dim).to(self.device)
        input_tensor = input_tensor.to(self.device)

        return self.test(softmax_ref, softmax_test, input_tensor)

    def test_harness(self, results, test_name, tester, args):
        results[test_name] = tester(**args)
        return results


    def run_timed_test(self, result, test_name, operation, expected_output, compare_op=lambda x, y: x == y):
        start_time = time.time()
        output = operation()
        end_time = time.time()
        result[test_name] = {
            "timed_test": {
                "pass": "PASS" if compare_op(output, expected_output) else "FAIL",
                "time": end_time - start_time,
                "output": output,
                "expected_output": expected_output
            }
        }
        return result

    def run_all_tests(self):
        results = {}
        set_seed(42)

        results = self.test_harness(
            results,
            "SoftMax Last Dim 10",
            self.test_softmax,
            {
                "input_tensor": torch.randn(1, 10),
                "dim": 1
            }
        )

        results = self.test_harness(
            results,
            "SoftMax Mutli Dimension",
            self.test_softmax,
            {
                "input_tensor": torch.randn(25, 5, 10),
                "dim": 1
            }
        )

        results = self.test_harness(
            results,
            "Timing Test 1",
            self.test_softmax,
            {
                "input_tensor": torch.randn(64, 1024) - 3,
                "dim": 1
            }
        )

        results = self.test_harness(
            results,
            "Timing Test 2",
            self.test_softmax,
            {
                "input_tensor": torch.randn(100, 1024) + 3,
                "dim": 1
            }
        )


        #//@@ HINT if you are running into issues here on GPU
        #//@@ Before anything else you should see if 
        #//@@ 65535 is an important number for 1080ti's...
        results = self.test_harness(
            results,
            "Timing Test 5 (Test some fun behavior in CUDA)",
            self.test_softmax,
            {
                "input_tensor": torch.randn(1, 65535 + 5),
                "dim": 1
            }
        )


        #//@@ Quick hint here: you are more likely to have numerical stability issues
        #//@@ With the LLms that will make softmax not sum up to 1 in each dimension
        self.run_timed_test(results, "GPT2", lambda: run_gpt_inference(), 
            "Hello, I'm a language model, I'm a problem solver.\n\nWhen I started translating, I was trying to solve my"
        )

        self.run_timed_test(results, "tinyllama", lambda: run_tinyllama_inference(), 
            "Hello, I'm a language model, and I can generate text in response to a given prompt.\n\nBased on the passage,"
        )


        #//@@ Qwen 3.5 0.8b was too big, if you want to try it go ahead and uncomment
        #//@@ Our testing with DSMLP had it take 20 mins running everything but softmax on cpu
        #//@@ But on a good cpu about 40 seconds so ¯\_(ツ)_/¯
        # self.run_timed_test(results, "qwen", lambda: run_qwen_inference(), 
        #     "Hello, I'm a language model, but I was asked to write a very basic AI assistant that is useful to users to learn and use"
        # )
        return results

    def find_failing_tests(self, results):
        failing_tests = []
        for test_name, test_result in results.items():
            for subtest_name, subtest_result in test_result.items():
                # if subtest_result["pass"] == "FAIL":
                print(
                    f"Test: {test_name}",
                    subtest_name,
                    json.dumps(test_result, indent=4), 
                    "\n_________________________________\n"
                )

        return failing_tests


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Tester for PA8",
        description="Profiles student code using nvprof.",
    )    
    
    parser.add_argument('--device_id', default="-1")
    parser.add_argument('--platform_id', default="-1") 
    parser.add_argument('--device_type', default="GPU")   
    parser.add_argument('--result_path', default="test_results.json")
    # Good luck trying to figure out the name of the json file we are writing to when grading   

    args = parser.parse_args()

    if args.device_id == "-1":
        args.device_id = None
    else:
        args.device_id = int(args.device_id)
    if args.platform_id == "-1":
        args.platform_id = None
    else:        
        args.platform_id = int(args.platform_id)

    # Pytorch_ocl doesn't play well with attention mechanisms anyway
    # if "pytorch_ocl" in os.environ:
    #     device = select_device(selected_plat_id=args.platform_id, selected_device_id=args.device_id, selected_device_type=args.device_type, return_all=False)
    # else:
    device = "cpu"

    tester = TestSoftmax(device=device)

    results = tester.run_all_tests()
    failing_tests = tester.find_failing_tests(results)

    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    if len(failing_tests) > 0:
        raise RuntimeError("Failed Test") 
        