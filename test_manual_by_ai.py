import os
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from operator import add
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, \
    PrecisionRecallDisplay
from model import build_unet
from functions import make_dir, seeding, write_metrics_report, generate_folder_name, get_image_keypoints


def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


def calculate_precision_recall_curve(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    PrecisionRecallDisplay.from_predictions(y_true, y_pred)
    plt.show()


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
    points_predicted_per_image = []
    time_taken = []

    report_name = generate_folder_name("results/")

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = f"image_mask_pred_{i}"

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

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        pred_y = pred_y * 255

        ai_segmentation = cv2.addWeighted(image, 1, pred_y, 1, 0)
        manual_segmentation = cv2.addWeighted(image, 1, ori_mask, 1, 0)

        keypoints = get_image_keypoints(pred_y)
        percentage_points_predicted = len(keypoints) / 18
        points_predicted_per_image.append(percentage_points_predicted)

        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            s = keypoint.size

            cv2.putText(ai_segmentation, f"({x}, {y})", (x - 10, y - 10), cv2.FONT_ITALIC, 0.2, color=(100, 100, 100))

        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [manual_segmentation, line, ai_segmentation], axis=1
        )

        make_dir(f"results/{report_name}")
        cv2.imwrite(f"results/{report_name}/{name}.png", cat_images)

    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    points_detected = sum(points_predicted_per_image) / 20

    metrics = f"Jaccard: {jaccard:1.4f}\nF1: {f1:1.4f}\nRecall: {recall:1.4f}\nPrecision: {precision:1.4f}\nAcc: {acc:1.4f}\nPoints: {points_detected:1.4f}"

    os.chdir(f"results/{report_name}")
    write_metrics_report(metrics)
