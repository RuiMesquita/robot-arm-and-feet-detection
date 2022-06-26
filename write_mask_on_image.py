import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from datetime import datetime
import torch

from model import build_unet
from functions import make_dir, seeding, generate_folder_name


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


if __name__ == "__main__":
    """ seeding """
    seeding(42)

    """ Folders """
    make_dir("../results")

    """ Load the dataset """
    test_x = sorted(glob("./data/test/images/*"))
    test_y = sorted(glob("./data/test/masks/*"))

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "target/model_1.3.0.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    report_name = generate_folder_name("results/")

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = f"mask_over_image{i}"

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))
        x = x/255.0
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)
        y = y/255.0
        y = np.expand_dims(y, axis=0)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)

            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        
        image_overlayed = cv2.addWeighted(image, 1, pred_y*255, 1, 0)
        
        make_dir(f"results/{report_name}")
        cv2.imwrite(f"results/{report_name}/{name}.png", image_overlayed)

    os.chdir(f"./results/{report_name}")
