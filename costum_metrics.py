import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import statistics


class customMetrics:
    def __init__(self, image):
        self.image = image
        self.points = []
        self.left_points = []
        self.right_points = []
        self.heel_points = []
        self.plantar_points = []
        self.finger_points = []

    def blob_filter(self):
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByColor = False
        params.filterByArea = True
        params.minArea = 5
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False

        inv_image = cv2.bitwise_not(self.image)
        detector = cv2.SimpleBlobDetector_create(params)

        self.points = detector.detect(inv_image)

    def get_left_feet_points(self):
        for point in self.points:
            x = int(point.pt[0])

            if x > self.image.shape[1] / 2:
                self.left_points.append(point)

        return len(self.left_points)

    def get_right_feet_points(self):
        for point in self.points:
            x = int(point.pt[0])

            if x < self.image.shape[1] / 2:
                self.right_points.append(point)

        return len(self.right_points)

    def get_feet_y_dimensions(self):
        y_list = [int(point.pt[1]) for point in self.points]

        min_y = min(y_list)
        max_y = max(y_list)
        foot_height = (max_y - min_y) + 10

        return min_y, max_y, foot_height

    def get_heel_points(self):
        min_y, max_y, foot_height = self.get_feet_y_dimensions()
        for point in self.points:
            y = point.pt[1]

            if y > min_y + (0.9 * foot_height):
                self.heel_points.append(point)

        return len(self.heel_points)

    def get_plantar_points(self):
        min_y, max_y, foot_height = self.get_feet_y_dimensions()
        for point in self.points:
            y = point.pt[1]

            if min_y + (0.9 * foot_height) > y > min_y + (foot_height * 0.4):
                self.plantar_points.append(point)

        return len(self.plantar_points)

    def get_finger_points(self):
        min_y, max_y, foot_height = self.get_feet_y_dimensions()
        for point in self.points:
            y = point.pt[1]

            if y < min_y + (foot_height * 0.4):
                self.finger_points.append(point)

        return len(self.finger_points)

    def get_all_points(self):
        return len(self.points)


def box_plotting(array):

    sns.boxplot(y=array)
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('./results/report004/mask_over_image0.png')

    # call class customMetrics
    metrics = customMetrics(img)

    # trigger the filter to get all the points
    metrics.blob_filter()

    # custom metrics
    print("All: ", metrics.get_all_points())
    print("Left", metrics.get_left_feet_points())
    print("Right", metrics.get_right_feet_points())
    print("Heel", metrics.get_heel_points())
    print("Plantar", metrics.get_plantar_points())
    print("Finger:", metrics.get_finger_points())

    y = [1, 2, 4, 6, 8, 3, 6, 3, 8, 2, 8, 3, 9, 3, 9, 3, 9]
    box_plotting(y)

    y = statistics.mean(y)
    print(y)



