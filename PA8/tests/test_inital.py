## DEPRECATED
# import time
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# from opencl_functions import ocl_conv2d
# # import transformers
# # from transformers import pipeline, set_seed

# class OclConv2d(nn.Conv2d):
#     def _conv_forward(self, input, weight, bias):
#         print(self.padding_mode)
#         print(self.stride, self.dilation, self.groups)
#         padding = self.padding
#         if self.padding_mode != "zeros":
#             input = F.pad(
#                 input, self._reversed_padding_repeated_twice, mode=self.padding_mode
#             )
#             padding =  _pair(0)
        
#         # return F.conv2d(
#         #     input, weight, bias, self.stride, padding, self.dilation, self.groups
#         # )
#         else:
#             # test cases with simple conv2d pass, so is it the padding?
#             input = F.pad(input, pad=(padding[1], padding[1], padding[0], padding[0]), mode='constant', value=0)

#         print(input.shape, weight.shape, bias, self.out_channels)

#         if bias is None:
#             bias = torch.zeros(weight.shape[0])

#         output = ocl_conv2d(
#             input,
#             weight,
#             bias,
#             self.stride[0],
#             self.stride[1],
#             self.dilation[0],
#             self.dilation[1],
#             self.groups
#         )
#         return output

# def compare_conv_outputs(conv_reference, conv_test, input_tensor, test_name):
#     """
#     Helper function to compare outputs from two conv layers.
    
#     Args:
#         conv_reference: Reference nn.Conv2d layer
#         conv_test: Test conv layer (subclass of nn.Conv2d)
#         input_tensor: Input tensor to test
#         test_name: Name of the test for reporting
#     """
#     # Ensure both use the same weights and bias
#     with torch.no_grad():
#         conv_test.weight.copy_(conv_reference.weight)
#         if conv_reference.bias is not None:
#             conv_test.bias.copy_(conv_reference.bias)
    
#     print(conv_test.weight)


#     # Get outputs
#     output_reference = conv_reference(input_tensor)
#     output_test = conv_test(input_tensor)
#     # print(output_reference, output_test)

#     # Compare
#     assert output_reference.shape == output_test.shape, \
#         f"{test_name}: Shape mismatch! Reference: {output_reference.shape}, Test: {output_test.shape}"
    
#     assert torch.allclose(output_reference, output_test, rtol=1e-5, atol=1e-5), \
#         f"{test_name}: Output mismatch!\nMax diff: {(output_reference - output_test).abs().max()}"
    
    
#     print(f"✓ {test_name} passed")
#     return True


# def test_basic_conv2d():
#     """Test basic 2D convolution with default parameters."""
#     input_tensor = torch.randn(1, 3, 5, 5)
    
#     conv_ref = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
#     conv_test = OclConv2d(in_channels=3, out_channels=2, kernel_size=3)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Basic conv2d")


# def test_conv2d_with_bias():
#     """Test 2D convolution with bias."""
#     input_tensor = torch.randn(1, 3, 5, 5)
    
#     conv_ref = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, bias=True)
#     conv_test = OclConv2d(in_channels=3, out_channels=2, kernel_size=3, bias=True)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with bias")


# def test_conv2d_no_bias():
#     """Test 2D convolution without bias."""
#     input_tensor = torch.randn(1, 3, 5, 5)
    
#     conv_ref = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, bias=False)
#     conv_test = OclConv2d(in_channels=3, out_channels=2, kernel_size=3, bias=False)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d without bias")


# def test_conv2d_with_stride():
#     """Test 2D convolution with custom stride."""
#     input_tensor = torch.randn(1, 3, 8, 8)
    
#     conv_ref = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=2)
#     conv_test = OclConv2d(in_channels=3, out_channels=2, kernel_size=3, stride=2)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with stride=2")


# def test_conv2d_with_padding():
#     """Test 2D convolution with padding."""
#     input_tensor = torch.randn(1, 3, 5, 5)
    
#     conv_ref = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1)
#     conv_test = OclConv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with padding=1")


# def test_conv2d_with_dilation():
#     """Test 2D convolution with dilation."""
#     input_tensor = torch.randn(1, 3, 7, 7)
    
#     conv_ref = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, dilation=2)
#     conv_test = OclConv2d(in_channels=3, out_channels=2, kernel_size=3, dilation=2)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with dilation=2")


# def test_conv2d_with_groups():
#     """Test grouped convolution."""
#     input_tensor = torch.randn(1, 4, 5, 5)
    
#     conv_ref = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, groups=2)
#     conv_test = OclConv2d(in_channels=4, out_channels=4, kernel_size=3, groups=2)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with groups=2")


# def test_conv2d_batch():
#     """Test 2D convolution with batch size > 1."""
#     batch_size = 4
#     input_tensor = torch.randn(batch_size, 3, 5, 5)
    
#     conv_ref = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
#     conv_test = OclConv2d(in_channels=3, out_channels=2, kernel_size=3)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with batch_size=4")


# def test_conv2d_1x1_kernel():
#     """Test 2D convolution with 1x1 kernel (pointwise convolution)."""
#     input_tensor = torch.randn(1, 3, 5, 5)
    
#     conv_ref = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)
#     conv_test = OclConv2d(in_channels=3, out_channels=8, kernel_size=1)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with 1x1 kernel")


# def test_conv2d_different_padding():
#     """Test 2D convolution with different padding for height and width."""
#     input_tensor = torch.randn(1, 3, 5, 5)
    
#     conv_ref = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=(1, 2))
#     conv_test = OclConv2d(in_channels=3, out_channels=2, kernel_size=3, padding=(1, 2))
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with padding=(1,2)")


# def test_conv2d_simple_values():
#     """Test conv2d with simple known values."""
#     input_tensor = torch.tensor([[[[1., 2., 3.],
#                                    [4., 5., 6.],
#                                    [7., 8., 9.]]]])
    
#     conv_ref = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
#     conv_test = OclConv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
    
#     with torch.no_grad():
#         conv_ref.weight = nn.Parameter(torch.tensor([[[[1., 0.],
#                                                         [0., 1.]]]]))
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with simple values")


# def test_conv2d_multichannel_simple():
#     """Test conv2d with multiple input channels and simple values."""
#     input_tensor = torch.tensor([[
#         [[1., 2., 3.],
#          [4., 5., 6.],
#          [7., 8., 9.]],
#         [[1., 1., 1.],
#          [1., 1., 1.],
#          [1., 1., 1.]]
#     ]])
    
#     conv_ref = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=2, bias=False)
#     conv_test = OclConv2d(in_channels=2, out_channels=1, kernel_size=2, bias=False)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d multichannel simple")


# def test_conv2d_stride_simple():
#     """Test conv2d with stride=2 and simple values."""
#     input_tensor = torch.tensor([[[[1., 2., 3., 4., 5.],
#                                    [6., 7., 8., 9., 10.],
#                                    [11., 12., 13., 14., 15.],
#                                    [16., 17., 18., 19., 20.],
#                                    [21., 22., 23., 24., 25.]]]])
    
#     conv_ref = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, bias=False)
#     conv_test = OclConv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, bias=False)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with stride=2 simple")


# def test_conv2d_padding_simple():
#     """Test conv2d with padding and simple values."""
#     input_tensor = torch.tensor([[[[1., 2., 3.],
#                                    [4., 5., 6.],
#                                    [7., 8., 9.]]]])
    
#     conv_ref = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
#     conv_test = OclConv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with padding simple")


# def test_conv2d_bias_simple():
#     """Test conv2d with bias and simple values."""
#     input_tensor = torch.tensor([[[[1., 2.],
#                                    [3., 4.]]]])
    
#     conv_ref = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=True)
#     conv_test = OclConv2d(in_channels=1, out_channels=1, kernel_size=2, bias=True)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d with bias simple")


# def test_conv2d_multiple_output_channels_simple():
#     """Test conv2d with multiple output channels and simple values."""
#     input_tensor = torch.tensor([[[[1., 2., 3.],
#                                    [4., 5., 6.],
#                                    [7., 8., 9.]]]])
    
#     conv_ref = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, bias=False)
#     conv_test = OclConv2d(in_channels=1, out_channels=2, kernel_size=2, bias=False)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d multiple output channels")


# def test_conv2d_asymmetric_kernel_simple():
#     """Test conv2d with non-square kernel and simple values."""
#     input_tensor = torch.tensor([[[[1., 2., 3., 4., 5.],
#                                    [6., 7., 8., 9., 10.],
#                                    [11., 12., 13., 14., 15.],
#                                    [16., 17., 18., 19., 20.]]]])
    
#     conv_ref = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 3), bias=False)
#     conv_test = OclConv2d(in_channels=1, out_channels=1, kernel_size=(2, 3), bias=False)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d asymmetric kernel")


# def test_conv2d_batch_simple():
#     """Test conv2d with batch size > 1 and simple values."""
#     input_tensor = torch.tensor([
#         [[[1., 2.],
#           [3., 4.]]],
#         [[[5., 6.],
#           [7., 8.]]]
#     ])
    
#     conv_ref = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
#     conv_test = OclConv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d batch simple")


# def test_conv2d_identity_kernel():
#     """Test conv2d with identity-like kernel."""
#     input_tensor = torch.tensor([[[[1., 2., 3.],
#                                    [4., 5., 6.],
#                                    [7., 8., 9.]]]])
    
#     conv_ref = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
#     conv_test = OclConv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
    
#     with torch.no_grad():
#         conv_ref.weight = nn.Parameter(torch.tensor([[[[0., 0., 0.],
#                                                         [0., 1., 0.],
#                                                         [0., 0., 0.]]]]))
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d identity kernel")


# def test_conv2d_zero_padding_effect():
#     """Test how zero padding affects edge computations."""
#     input_tensor = torch.tensor([[[[1., 2.],
#                                    [3., 4.]]]])
    
#     conv_ref = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, 
#                          padding_mode='zeros', bias=False)
#     conv_test = OclConv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, 
#                           padding_mode='zeros', bias=False)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, "Conv2d zero padding effect")


# def test_conv2d_large_random():
#     """
#     Test conv2d with large random inputs.
    
#     This tests the implementation with realistic sizes:
#     - Multiple batches
#     - Many channels
#     - Larger spatial dimensions
#     - Various kernel sizes and parameters
#     """
#     print("\n=== Large Random Tests ===")
    
#     # Test 1: Large batch with many channels
#     batch_size = 1
#     in_channels = 2 #Issue with in_channels somewhere
#     out_channels = 1
#     input_size = 100
#     kernel_size = 3
    
#     input_tensor = torch.randn(batch_size, in_channels, input_size, input_size)
    
#     conv_ref = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
#                          kernel_size=kernel_size, padding=1, bias=False)
#     conv_test = OclConv2d(in_channels=in_channels, out_channels=out_channels, 
#                           kernel_size=kernel_size, padding=1, bias=False)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, 
#                         f"Large random: batch={batch_size}, channels={in_channels}->{out_channels}, size={input_size}x{input_size}")
    
#     # Test 2: Large spatial dimensions with stride
#     batch_size = 8
#     in_channels = 16
#     out_channels = 32
#     input_size = 256
#     kernel_size = 5
#     stride = 2
    
#     input_tensor = torch.randn(batch_size, in_channels, input_size, input_size)
    
#     conv_ref = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
#                          kernel_size=kernel_size, stride=stride, padding=2, bias=True)
#     conv_test = OclConv2d(in_channels=in_channels, out_channels=out_channels, 
#                           kernel_size=kernel_size, stride=stride, padding=2, bias=True)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, 
#                         f"Large random with stride: batch={batch_size}, size={input_size}x{input_size}, stride={stride}")
    
#     # Test 3: Very deep (many channels) but smaller spatial
#     batch_size = 32
#     in_channels = 128
#     out_channels = 256
#     input_size = 64
#     kernel_size = 3
    
#     input_tensor = torch.randn(batch_size, in_channels, input_size, input_size)
    
#     conv_ref = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
#                          kernel_size=kernel_size, padding=1, bias=False)
#     conv_test = OclConv2d(in_channels=in_channels, out_channels=out_channels, 
#                           kernel_size=kernel_size, padding=1, bias=False)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, 
#                         f"Deep network: batch={batch_size}, channels={in_channels}->{out_channels}")
    
#     # Test 4: Large kernel size
#     batch_size = 4
#     in_channels = 8
#     out_channels = 16
#     input_size = 200
#     kernel_size = 7
    
#     input_tensor = torch.randn(batch_size, in_channels, input_size, input_size)
    
#     conv_ref = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
#                          kernel_size=kernel_size, padding=3, bias=True)
#     conv_test = OclConv2d(in_channels=in_channels, out_channels=out_channels, 
#                           kernel_size=kernel_size, padding=3, bias=True)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, 
#                         f"Large kernel: kernel_size={kernel_size}x{kernel_size}, size={input_size}x{input_size}")
    
#     # Test 5: Grouped convolution with larger inputs
#     batch_size = 8
#     in_channels = 64
#     out_channels = 64
#     groups = 1
#     input_size = 112
#     kernel_size = 3
    
#     input_tensor = torch.randn(batch_size, in_channels, input_size, input_size)
    
#     conv_ref = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
#                          kernel_size=kernel_size, groups=groups, padding=1, bias=True)
#     conv_test = OclConv2d(in_channels=in_channels, out_channels=out_channels, 
#                           kernel_size=kernel_size, groups=groups, padding=1, bias=True)
    
#     compare_conv_outputs(conv_ref, conv_test, input_tensor, 
#                         f"Grouped convolution: groups={groups}, size={input_size}x{input_size}")


# def run_all_tests():
#     """Run all test cases."""
#     print("\n" + "="*70)
#     print("Testing Conv2d Subclasses Against nn.Conv2d")
#     print("="*70 + "\n")
    
#     print("=== Basic Functionality Tests ===")
#     test_basic_conv2d()
#     test_conv2d_with_bias()
#     test_conv2d_no_bias()
#     test_conv2d_with_stride()
#     test_conv2d_with_padding()
#     test_conv2d_with_dilation()
#     # test_conv2d_with_groups()
#     test_conv2d_batch()
#     test_conv2d_1x1_kernel()
#     test_conv2d_different_padding()
    
#     print("\n=== Simple Values Tests ===")
#     test_conv2d_simple_values()
#     test_conv2d_multichannel_simple()
#     test_conv2d_stride_simple()
#     test_conv2d_padding_simple()
#     test_conv2d_bias_simple()
#     test_conv2d_multiple_output_channels_simple()
#     test_conv2d_asymmetric_kernel_simple()
#     test_conv2d_batch_simple()
#     test_conv2d_identity_kernel()
#     test_conv2d_zero_padding_effect()
    
#     print("\n=== Large Random Tests ===")
#     test_conv2d_large_random()
    
#     print("\n" + "="*70)
#     print("✅ All tests passed!")
#     print("="*70 + "\n")


# if __name__ == "__main__":
#     run_all_tests()
