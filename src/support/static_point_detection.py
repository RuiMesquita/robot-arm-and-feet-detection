import cv2
import numpy as np
import functions as fnc
import torch

from model import build_unet
from glob import glob
from tqdm import tqdm


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


if __name__ == "__main__":
    """ seeding """
    fnc.seeding(42)

    """ Folders """
    fnc.make_dir("../reports")

    """ Load the dataset """
    test_x = sorted(glob("../data/test/images/*"))
    test_y = sorted(glob("../data/test/masks/*"))

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "../target/unet_9p_50.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    report_name = fnc.generate_folder_name("../reports/")

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
            pred_y = pred_y*255
            
            image_overlayed = cv2.addWeighted(image, 1, pred_y, 1, 0)

            keypoints = fnc.get_image_keypoints(pred_y)

            for keypoint in keypoints:
                x = int(keypoint.pt[0])
                y = int(keypoint.pt[1])
                s = keypoint.size

                cv2.putText(image_overlayed, f"({x}, {y})", (x-10, y-10), cv2.FONT_ITALIC, 0.2, color=(100, 100, 100))

            fnc.make_dir(f'reports/{report_name}')
            cv2.imwrite(f"reports/{report_name}/{name}.png", image_overlayed)
