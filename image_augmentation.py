import cv2
from imgaug import augmenters as iaa
import os

# Path to the image folder
image_folder = "/Users/ljp176/Downloads/faces"

# Define augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # random crops
    iaa.Sometimes(0.5,
                  iaa.GaussianBlur(sigma=(0, 0.5))
                 ),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        rotate=(-25, 25),
    )
], random_order=True)  # apply augmenters in random order

for filename in os.listdir(image_folder):
    if filename.endswith(".jpeg"):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        image_aug = seq.augment_image(image)
        cv2.imwrite(image_path.replace(".jpeg", "_aug1.jpeg"), image_aug)
