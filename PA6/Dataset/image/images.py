import numpy as np
from PIL import Image

def get_legally_distant_character():
    pic = Image.open("Dataset/image/legally_distant_charater.png")
    return np.array(pic)

def get_pet_photos():
    iamges = [
            "Dataset/image/cat1.png",
            "Dataset/image/cat2.png",
            "Dataset/image/dog1.png",
            "Dataset/image/dog2.png",
            "Dataset/image/not_dog.png"
    ]

    labels = [
        'hamper',
        'Persian cat',
        'Labrador retriever',
        'malinois',
        'Siamese cat, Siamese'
    ]

    for image, label in zip(iamges, labels):
        yield image, label