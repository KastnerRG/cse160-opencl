import time
import json
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import torch.nn as nn
from opencl_functions import ocl_conv2d
import pytorch_ocl
import numpy as np
import os
from transformers import set_seed

from utils.select_device import select_device
from Dataset.image.images import get_legally_distant_character, get_pet_photos
from inferance_resnet import run_resnet_inference
from inferance_bird import run_bird_inference
from inferance_gpt2 import run_gpt_inference

class OclConv2d(nn.Conv2d):
    """A custom implementation of Conv2d that uses OpenCL for convolution operations."""
    def _conv_forward(self, input, weight, bias):
        """Override the _conv_forward method to use OpenCL for convolution.
        
        Args:            
            input (torch.Tensor): The input tensor.
            weight (torch.Tensor): The convolution weights.
            bias (torch.Tensor or None): The convolution bias. For this assignment, assume bias is always a tensor. 
        Returns:
            torch.Tensor: The result of the convolution operation.
        """
        padding = self.padding
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
            padding =  _pair(0)

        else:
            input = F.pad(input, pad=(padding[1], padding[1], padding[0], padding[0]), mode='constant', value=0)

        # In pytorch, Bias can be set to none (so the output of one convolution is just w^Tx)
        # to make things easier we can just set it to zero if there is no bias
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

class TestConv2d:
    """A class to test the correctness of the OclConv2d implementation."""
    def __init__(self, device):
        self.device = device


    def test(self, conv_reference, conv_test, input_tensor):
        result = {}
        with torch.no_grad():
            conv_test.weight.copy_(conv_reference.weight)
            if conv_reference.bias is not None:
                conv_test.bias.copy_(conv_reference.bias)
        
        output_reference = conv_reference(input_tensor)
        output_test = conv_test(input_tensor)

        result["shape"] = {
            "pass": "PASS" if output_reference.shape == output_test.shape else "FAIL",
            "reference": output_reference.shape,
            "student_implementation": output_test.shape  
        }
        
        result["value"] = {                     # Pytorch_OCL doesn't support allclose :(
            "pass": "PASS" if torch.allclose(output_reference.cpu(), output_test.cpu(), rtol=1e-5, atol=1e-5) else "FAIL"
        }
       
        return result

    def test_return_output(self, conv_reference, conv_test, input_tensor):
        with torch.no_grad():
            conv_test.weight.copy_(conv_reference.weight)
            if conv_reference.bias is not None:
                conv_test.bias.copy_(conv_reference.bias)
        
        output_reference = conv_reference(input_tensor)
        output_test = conv_test(input_tensor)
        
        return output_reference, output_test
       

    def test_random_weights_harness(self, input_tensor, in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1):
        """Test the forward pass of the OclConv2d layer using randomly initalized weights."""
        conv_reference = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1).to(self.device)
        conv_test = OclConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1).to(self.device)
        input_tensor = input_tensor.to(self.device)
        return self.test(conv_reference, conv_test, input_tensor)

    def test_specific_weights_harness(self, input_tensor, weight, bias, in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, testing_func=None):
        """Test the forward pass of the OclConv2d layer using specific weights and bias."""
        conv_reference = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1)
        conv_test = OclConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1)
        
        with torch.no_grad():
            conv_reference.weight.copy_(weight)
            conv_reference.bias.copy_(bias)
            conv_test.weight.copy_(weight)
            conv_test.bias.copy_(bias)

        conv_reference = conv_reference.to(self.device)
        conv_test = conv_test.to(self.device)
        input_tensor = input_tensor.to(self.device)
        if testing_func is None:
            testing_func = self.test
        return testing_func(conv_reference, conv_test, input_tensor)

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

        charater_from_beloved_cse160 = get_legally_distant_character()
        identity_kernel = torch.tensor(np.eye(3).reshape(1, 1, 3, 3))
        identity_kernel_per_channel = torch.tensor(np.concatenate([np.eye(3).reshape(1, 1, 3, 3)] * 4, axis=1))
        left_edge_detection_kernel = torch.tensor(np.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]]))
        edge_detection_kernel = torch.tensor(np.array([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]))

        set_seed(42)

        ## Normal Convolution (With batching)
        results = self.test_harness(
            results,
            "Normal Convolution Test 1",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 1, 224, 224),
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
             "Normal Convolution Test 2",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 3, 224, 224),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
             "Normal Convolution Test 3",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 3, 224, 224),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 9,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Normal Convolution Test 4",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 3, 512, 512),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Normal Convolution Test 5",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 3, 512, 512),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 5,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Normal Convolution Test 6",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 3, 512, 512),
                "in_channels": 3,
                "out_channels": 5,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Normal Convolution Test 7 (Idenity Matrix)",
            self.test_specific_weights_harness,
            {
                "input_tensor": torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
                "weight": identity_kernel_per_channel.float(),
                "bias": torch.zeros(1).float(), 
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )


        results = self.test_harness(
            results,
            "Normal Convolution Test 8 (Edge Dection)",
            self.test_specific_weights_harness,
            {
                "input_tensor": torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
                "weight": edge_detection_kernel.float(),
                "bias": torch.zeros(1).float(), 
                "in_channels": 4,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Normal Convolution Test 9 (Edge Dection)",
            self.test_specific_weights_harness,
            {
                "input_tensor": torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
                "weight": left_edge_detection_kernel.float(),
                "bias": torch.zeros(4).float(), 
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Normal Convolution Test 10 (Edge Dection)",
            self.test_specific_weights_harness,
            {
                "input_tensor": torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
                "weight": edge_detection_kernel.float(),
                "bias": torch.zeros(4).float(), 
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )



        ## Strided test Cases
        results = self.test_harness(
            results,
            "Strided Test 1",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 1, 224, 224),
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Strided Test 2",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 1, 224, 224),
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 3,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Strided Test 3",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 1, 224, 224)-3,
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": (3, 2),
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Strided Test 4",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 3, 512, 512),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 5,
                "stride": 2,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Strided Test 5",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 3, 512, 512),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 5,
                "stride": (3, 4),
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Strided Test 6",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 3, 512, 512),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 5,
                "stride": 5,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Strided Test 7",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 3, 512, 512) + 3,
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 5,
                "stride": 5,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Strided Test 8 (Idenity Matrix)",
            self.test_specific_weights_harness,
            {
                "input_tensor": torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
                "weight": identity_kernel_per_channel.float(),
                "bias": torch.zeros(1).float(), 
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1
            }
        )

        print(left_edge_detection_kernel.shape, torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).shape)

        results = self.test_harness(
            results,
            "Strided Test 9 (Edge Detection)",
            self.test_specific_weights_harness,
            {
                "input_tensor": torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
                "weight": left_edge_detection_kernel.float(),
                "bias": torch.zeros(4).float(), 
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )

        results = self.test_harness(
            results,
            "Strided Test 10 (Edge Detection)",
            self.test_specific_weights_harness,
            {
                "input_tensor": torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
                "weight": edge_detection_kernel.float(),
                "bias": torch.zeros(4).float(), 
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1
            }
        )


        ## Dialation
        results = self.test_harness(
            results,
            "Dilated Test 1",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 1, 224, 224),
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 2
            }
        )

        results = self.test_harness(
            results,
            "Dilated Test 2",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 1, 224, 224),
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 4,
                "stride": 1,
                "padding": 1,
                "dilation": 2
            }
        )

        results = self.test_harness(
            results,
            "Dilated Test 3",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 1, 224, 224)-3,
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": (3, 1)
            }
        )

        results = self.test_harness(
            results,
            "Dilated Test 4",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(5, 3, 512, 512),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 5,
                "stride": 1,
                "padding": 1,
                "dilation": (1, 3)
            }
        )

        results = self.test_harness(
            results,
            "Dilated Test 5",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 3, 512, 512),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 5,
                "stride": 1,
                "padding": 1,
                "dilation": (2, 3)
            }
        )

        results = self.test_harness(
            results,
            "Dilated Test 6",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 3, 512, 512),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 4
            }
        )

        results = self.test_harness(
            results,
            "Dilated Test 7",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(4, 3, 512, 512) + 3,
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 5,
                "stride": 1,
                "padding": 1,
                "dilation": 3
            }
        )

        results = self.test_harness(
            results,
            "Dilated Test 8 (Idenity Matrix)",
            self.test_specific_weights_harness,
            {
                "input_tensor": torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
                "weight": identity_kernel_per_channel.float(),
                "bias": torch.zeros(1).float(), 
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 2
            }
        )

        results = self.test_harness(
            results,
            "Dilated Test 9 (Edge Dection)",
            self.test_specific_weights_harness,
            {
                "input_tensor": torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
                "weight": left_edge_detection_kernel.float(),
                "bias": torch.zeros(4).float(), 
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 3
            }
        )

        results = self.test_harness(
            results,
            "Dilated Test 10 (Edge Dection)",
            self.test_specific_weights_harness,
            {
                "input_tensor": torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
                "weight": edge_detection_kernel.float(),
                "bias": torch.zeros(4).float(), 
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 2
            }
        )

        ## Strided and Dilation
        results = self.test_harness(
            results,
            "Full Test 1",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 1, 224, 224),
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 2
            }
        )

        results = self.test_harness(
            results,
            "Full Test 2",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 1, 224, 224),
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 4,
                "stride": 3,
                "padding": 1,
                "dilation": 2
            }
        )

        results = self.test_harness(
            results,
            "Full Test 3",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(2, 1, 224, 224)-3,
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 3,
                "padding": 1,
                "dilation": 3
            }
        )

        results = self.test_harness(
            results,
            "Full Test 4",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(5, 3, 512, 512),
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 5,
                "stride": 3,
                "padding": 1,
                "dilation": 3
            }
        )

        results = self.test_harness(
            results,
            "Full Test 5",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(1, 7, 500, 707) + 4,
                "in_channels": 7,
                "out_channels": 6,
                "kernel_size": 11,
                "stride": (4, 8),
                "padding": 4,
                "dilation": (4, 3)
            }
        )

        results = self.test_harness(
            results,
            "Full Test 6",
            self.test_random_weights_harness,
            {
                "input_tensor": ((torch.randn(1, 7, 200, 200)) - 4),
                "in_channels": 7,
                "out_channels": 3,
                "kernel_size": 5,
                "stride": (3, 2),
                "padding": 4,
                "dilation": (3, 5)
            }
        )
       
        results = self.test_harness(
            results,
            "Full Test 7",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 12, 224, 224),
                "in_channels": 12,
                "out_channels": 24,
                "kernel_size": 5,
                "stride": 4,
                "padding": 4,
                "dilation": 4
            }
        )

        results = self.test_harness(
            results,
            "Full Test 8",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 12, 224, 224),
                "in_channels": 12,
                "out_channels": 24,
                "kernel_size": 5,
                "stride": (4, 3),
                "padding": 4,
                "dilation": (3, 4)
            }
        )

        results = self.test_harness(
            results,
            "Full Test 9",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 4, 224, 224),
                "in_channels": 4,
                "out_channels": 13,
                "kernel_size": 8,
                "stride": (4, 3),
                "padding": 4,
                "dilation": (3, 4)
            }
        )

        results = self.test_harness(
            results,
            "Full Test 10",
            self.test_random_weights_harness,
            {
                "input_tensor": torch.randn(3, 4, 224, 224),
                "in_channels": 4,
                "out_channels": 13,
                "kernel_size": 3,
                "stride": (5),
                "padding": 4,
                "dilation": (1, 2)
            }
        )

        ## Model Tests
        self.run_timed_test(results, "run inferance_gp2.py", lambda: run_gpt_inference(), "Hello, I'm a language model, I'm a problem solver.\n\nWhen I started translating, I was trying to solve my") # FIX TO COMPARE OUTPUT TEXT
        self.run_timed_test(results, "run inference_resnet.py", lambda: run_resnet_inference( image_path=""), "tiger cat")

        pet_photos_generator = get_pet_photos()
        while True:
            try:
                image_path, label = next(pet_photos_generator)
                self.run_timed_test(results, f"run_inference_pet_{len(results)}", lambda: run_resnet_inference(device=self.device, image_path=image_path), label)
            except StopIteration:
                break

        self.run_timed_test(results, "bird_classification", lambda: run_bird_inference(device=self.device), [
            78,
            104,
            108,
            104,
            104,
            22
        ], compare_op=lambda x, y: np.array_equal(x, y))

        return results

    def find_failing_tests(self, results):
        failing_tests = []
        for test_name, test_result in results.items():
            for subtest_name, subtest_result in test_result.items():
                if subtest_result["pass"] == "FAIL":
                    print(
                        "Test Failed:",
                        test_name,
                        subtest_name,
                        json.dumps(test_result, indent=4), 
                        "\n_________________________________\n"
                    )
        
        print("All tests completed.")
        return failing_tests


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Tester for PA6",
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

    if not "timing_mode" in os.environ:
        device = select_device(selected_plat_id=args.platform_id, selected_device_id=args.device_id, selected_device_type=args.device_type, return_all=False)
    else:
        device = "cpu"

    tester = TestConv2d(device=device)

    results = tester.run_all_tests()
    tester.find_failing_tests(results)

    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

        