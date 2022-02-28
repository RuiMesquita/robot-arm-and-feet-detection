import os
from glob import glob

import cv2
import imageio
import numpy as np
from albumentations import HorizontalFlip, Rotate, VerticalFlip
from tqdm import tqdm

import functions as fnc


def load_data(path):
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.JPG")))
    train_y = sorted(glob(os.path.join(path, "training", "2nd_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.JPG")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)


def data_augment(images, masks, save_path, augment=True):
    size = (512, 512)

    index = 0
    for i, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        # name of augmented images
        name = "aug"

        # read image
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        print(x.shape, y.shape)

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
            
        else:
            X = [x]
            Y = [y]
            

        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            print(index)
            image_name = f'{name}_image_{index}.png'
            mask_name = f'{name}_mask_{index}.png'

            image_path = os.path.join(save_path, "images/", image_name)
            mask_path = os.path.join(save_path, "masks/", mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    np.random.seed(42)

    """ Load Data """
    (train_x, train_y), (test_x, test_y) = load_data('./feet_data')

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Crete New Directories for augmented data"""
    fnc.make_dir("augmented_data/train/images/")
    fnc.make_dir("augmented_data/train/masks/")
    fnc.make_dir("augmented_data/test/images/")
    fnc.make_dir("augmented_data/test/masks/")

    """ Augment data """
    data_augment(train_x, train_y, "augmented_data/train/")
    data_augment(test_x, test_y, "augmented_data/test/", augment=False)
