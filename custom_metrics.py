import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os


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

            bot = int(min_y + (0.9 * foot_height))
            top = int(min_y + (foot_height * 0.4))

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


def box_plotting(dataframe):
    
    sns.boxplot(data=dataframe)
    plt.show()


if __name__ == '__main__':

    total_points = []
    left_points = []
    right_points = []
    heel_points = []
    plantar_points = []
    finger_points = []

    for filename in os.listdir("results/report004"):
        img = cv2.imread(os.path.join("results/report004", filename))

        # call class customMetrics
        metrics = customMetrics(img)

        # trigger the filter to get all the points
        metrics.blob_filter()

        # custom metrics
        all_p = metrics.get_all_points()
        left_p = metrics.get_left_feet_points()
        right_p = metrics.get_right_feet_points()
        heel_p = metrics.get_heel_points()
        plantar_p = metrics.get_plantar_points()
        finger_p = metrics.get_finger_points()

        print("All:", all_p)
        print("Left:", left_p)
        print("Right:", right_p)
        print("Heel:", heel_p)
        print("Plantar:", plantar_p)
        print("Finger:", finger_p, "\n")

        total_points.append(all_p)
        left_points.append(left_p)
        right_points.append(right_p)
        heel_points.append(heel_p)
        plantar_points.append(plantar_p)
        finger_points.append(finger_p)

        labels = ["All", "Left", "Right", "Heel", "Plantar", "Finger"]

    d = {'All': total_points, 'Left': left_points, 'Right': right_points, 'Heel': heel_points,
         'Plantar': plantar_points, "Finger": finger_points}
    df = pd.DataFrame(data=d)
    print(df)

    box_plotting(df)
