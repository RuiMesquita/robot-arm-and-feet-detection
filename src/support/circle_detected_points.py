import os
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from operator import add
from glob import glob
from tqdm import tqdm
from model import build_unet
from functions import seeding, get_image_keypoints


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


if __name__ == "__main__":
    """ seeding """
    seeding(42)

    """ Load the dataset """
    test_x = sorted(glob("./data/test/images/aug_image_0.png"))
    test_y = sorted(glob("./data/test/masks/aug_mask_0.png"))

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "target/unet_9p_50.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)
        y = y / 255.0
        y = np.expand_dims(y, axis=0)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        pred_y = pred_y * 255

        print(pred_y)

        keypoints = get_image_keypoints(pred_y)
        im_with_keypoints = cv2.drawKeypoints(pred_y, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)

        print("\n Detected Points:", len(keypoints))



