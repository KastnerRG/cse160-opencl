from test import TestConv2d
from utils.select_device import select_device
from Dataset.image.images import get_legally_distant_character
from PIL import Image
import torch
import numpy as np

def normalize_image(image):
    image = image - image.min()
    image = (image / image.max()) * 255
    return image.astype(np.uint8)

if __name__ == "__main__":
    device = select_device()
    tester = TestConv2d(device=device)

    charater_from_beloved_cse160 = get_legally_distant_character()
    # # identity_kernel = torch.tensor(np.eye(3).reshape(1, 1, 3, 3))
    # # identity_kernel_per_channel = torch.tensor(np.concatenate([np.eye(3).reshape(1, 1, 3, 3)] * 4, axis=1))
    # edge_detection_kernel = torch.tensor(np.array([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]))
    left_edge_detection_kernel = torch.tensor(np.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]]))

    with torch.no_grad():
        output_reference, output_test =tester.test_specific_weights_harness(
            input_tensor=torch.tensor(charater_from_beloved_cse160).reshape(1, 4, 32, 32).float(),
            weight=left_edge_detection_kernel.float(),
            bias=torch.zeros(4).float(), 
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            testing_func=tester.test_return_output
        )


    output_reference_image = normalize_image(output_reference.cpu().numpy().squeeze().transpose(1, 2, 0)[..., :3])
    output_test_image = normalize_image(output_test.cpu().numpy().squeeze().transpose(1, 2, 0)[..., :3])
    output_reference_image = Image.fromarray(output_reference_image).convert("RGB")
    output_test_image = Image.fromarray(output_test_image).convert("RGB")

    output_reference_image.save("output_reference.png")
    output_test_image.save("output_test.png")

