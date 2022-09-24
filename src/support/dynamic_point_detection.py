import cv2
import torch
import numpy as np
import functions as fnc

from model import build_unet


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame by frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end ?). Exiting ...")
        break

    # Operations on frame
    image = cv2.resize(frame, size)
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

    # Overlay the mask on top of the image
    image_overlayed = cv2.addWeighted(image, 1, pred_y, 1, 0)

    # Create annotations
    keypoints = fnc.get_image_keypoints(pred_y)

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        s = keypoint.size

        cv2.putText(image_overlayed, f"({x}, {y})", (x-10, y-10), cv2.FONT_ITALIC, 0.3, color=(255, 255, 255))


    # Display the resulting frame
    cv2.imshow('video', image_overlayed)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()