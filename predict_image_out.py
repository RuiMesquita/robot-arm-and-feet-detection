import cv2
import numpy as np
import functions as fnc
import torch

from model import build_unet


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


if __name__ == "__main__":
    """ seeding """
    fnc.seeding(42)

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

    image = cv2.imread("./aug_image_1.png", cv2.IMREAD_COLOR)
    image = cv2.resize(image, size)
    x = np.transpose(image, (2, 0, 1))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)

    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)

        pred_y = pred_y[0].cpu().numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

    pred_y = mask_parse(pred_y)
    pred_y = pred_y * 255

    image_overlayed = cv2.addWeighted(image, 1, pred_y, 1, 0)

    keypoints = fnc.get_image_keypoints(pred_y)

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        s = keypoint.size

        cv2.putText(image_overlayed, f"({x}, {y})", (x - 10, y - 10), cv2.FONT_ITALIC, 0.2, color=(100, 100, 100))

    cv2.imshow('prediction', image_overlayed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
