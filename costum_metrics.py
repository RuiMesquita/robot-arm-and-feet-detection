from typing_extensions import Self
import cv2


class CostumMetrics:
    def __init__(self, image):
        self.image = image
        self.points = []
        self.left_points = []
        self.right_points = []
    

    def blob_filter(self):
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByColor = False
        params.filterByArea = True
        params.minArea = 4
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False

        inv_image = cv2.bitwise_not(self.image)
        detector = cv2.SimpleBlobDetector_create(params)

        self.points = detector.detect(inv_image)


    def get_all_points(self):
        return len(self.points)

    
    def get_right_and_left_feet_points(self):
        for point in self.points:
            x = int(point.pt[0])

            if x < self.image.shape[2] / 2:
                self.left_points.append(point)
            else:
                self.right_points.append(point)


if __name__ == '__main__':

    img = cv2.imread('./results/report004/mask_over_image0.png')

    metrics = CostumMetrics(img) 

    metrics.blob_filter()
    metrics.get_right_and_left_feet_points()

    print("All: ", metrics.get_all_points())
    print("Left", len(metrics.left_points))
    print("Right", len(metrics.right_points))