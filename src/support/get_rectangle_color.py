import cv2 as cv2
import numpy as np

#load the image
img = cv2.imread("./data/test/images/aug_image_4.png")


# convert to hsv colorspace
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower bound and upper bound for yellow color
lower_bound = np.array([20, 80, 80])   
upper_bound = np.array([40, 255, 255])

# find the colors within the boundaries
mask = cv2.inRange(hsv, lower_bound, upper_bound)

#define kernel size  
kernel = np.ones((7,7),np.uint8)

# Remove unnecessary noise from mask
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

segmented_img = cv2.bitwise_and(img, img, mask=mask)

contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

#x,y,w,h = cv2.boundingRect(contours[0])


# Showing the output
cv2.imshow("HSV", hsv)
cv2.imshow("Mask", mask)
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

