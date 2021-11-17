import os 
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.JPG")))
    train_y = sorted(glob(os.path.join(path, "training", "2nd_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.JPG")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)

def data_augment(images, masks, save_path, augment=True):
    size = (512, 512)

    for i, (x, y) in tqdm(enumerate(zip(images, masks)), total = len(images)):
        # name of augmented images
        name = "aug_img"


if __name__== "__main__":
    np.random.seed(42)

    """ Load Data """
    (train_x, train_y), (test_x, test_y) = load_data('./feet_data')

    """ Crete New Directories for augmented data"""
    os.mkdir("augmented_data/train/image/")
    os.mkdir("augmented_data/train/mask/")
    os.mkdir("augmented_data/test/image/")
    os.mkdir("augmented_data/test/mask/")

    """ Augment data """
    data_augment()