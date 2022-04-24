from turtle import color
import numpy as np
import cv2

image_path = "./data/test/masks/aug_mask_2.png"
points = []

im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
label = "(x, y)"

detector = cv2.SimpleBlobDetector_create()

keypoints = detector.detect(cv2.bitwise_not(im))

#im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

for keypoint in keypoints:
    x = int(keypoint.pt[0])
    y = int(keypoint.pt[1])
    s = keypoint.size

    cv2.putText(im, f"({x}, {y})", (x-10, y-10), cv2.FONT_ITALIC, 0.2, color=(100, 100, 100))

cv2.imshow("Keypoints", im)
cv2.waitKey(0)
