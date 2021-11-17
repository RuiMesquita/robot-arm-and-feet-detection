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

    print(train_x)
    print(train_y)


if __name__== "__main__":
    np.random.seed(42)

    """ Load Data """
    load_data('./feet_data')