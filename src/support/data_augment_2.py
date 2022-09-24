import os
import functions as fnc
import cv2
import imageio
import numpy as np
import albumentations as alb

from tqdm import tqdm
from glob import glob


def load_data(path):
    train_x = sorted(glob(os.path.join(path, "train", "images", "*.JPG")))
    train_y = sorted(glob(os.path.join(path, "train", "binary_masks", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.JPG")))
    test_y = sorted(glob(os.path.join(path, "test", "binary_masks", "*.gif")))

    return (train_x, train_y), (test_x, test_y)


def data_augment(images, masks, save_path, augment=True):
    size = (512, 512)

    index = 0
    for i, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = "aug"

        # read image
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        # increase dataset by performing image and mask manipulations
        if augment:
            aug = alb.HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = alb.VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = alb.Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            aug = alb.GaussianBlur(blur_limit=(3, 7), p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]

            aug = alb.RandomBrightnessContrast(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            aug = alb.RGBShift(p=1.0)
            augmented = aug(image=x, mask=y)
            x6 = augmented["image"]
            y6 = augmented["mask"]

            aug = alb.ElasticTransform(p=1.0)
            augmented = aug(image=x, mask=y)
            x7 = augmented["image"]
            y7 = augmented["mask"]

            aug = alb.HueSaturationValue(always_apply=True)
            augmented = aug(image=x, mask=y)
            x8 = augmented["image"]
            y8 = augmented["mask"]

            X = [x, x1, x2, x3, x4, x5, x6, x7, x8]
            Y = [y, y1, y2, y3, y4, y5, y6, y7, y8]

        else:
            X = [x]
            Y = [y]

        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            image_name = f'{name}_image_{index}.png'
            mask_name = f'{name}_mask_{index}.png'

            image_path = os.path.join(save_path, "images", image_name)
            mask_path = os.path.join(save_path, "masks", mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    np.random.seed(42)

    """ Load Data """
    (train_x, train_y), (test_x, test_y) = load_data('./feet_data_9points')

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Crete New Directories for augmented data"""
    fnc.make_dir("data/train/images/")
    fnc.make_dir("data/train/masks/")
    fnc.make_dir("data/test/images/")
    fnc.make_dir("data/test/masks/")

    """ Augment data """
    data_augment(train_x, train_y, "data/train/")
    data_augment(test_x, test_y, "data/test/", augment=False)
