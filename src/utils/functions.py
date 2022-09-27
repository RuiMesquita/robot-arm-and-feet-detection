import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

""" Create dir if not exists """


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


""" Seeding the randomness """


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


""" calculate the time taken """


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


""" Save epoch progression to txt file """


def write_training_report(epoch_status):
    file = open("training_report.txt", "a")
    file.write(epoch_status)
    file.write("\n")
    file.close()


""" Store metrics report for each test """


def write_metrics_report(metrics):
    file = open("metrics.txt", "a")
    file.write(metrics)
    file.close()


""" Generate train loss/val loss graph """


def generate_graph_report(num_epochs, train_loss, valid_loss):
    x = list(range(1, num_epochs + 1))

    plt.plot(x, train_loss, color='r', label='Train Loss')
    plt.plot(x, valid_loss, color='g', label='Valid Loss')

    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Training report | Training Loss vs Validation Loss")

    plt.legend()
    plt.savefig('training_graph.png', bbox_inches='tight')
    plt.show()


""" Generate a folder name based in the existing folders """


def generate_folder_name(folder):
    number_of_dirs = 0
    for base, dirs, files in os.walk(folder):
        print("Searching in:", base)
        for directories in dirs:
            number_of_dirs += 1
            if str(number_of_dirs) in directories:
                number_of_dirs += 1

    return f"report{number_of_dirs:03}"


""" Blob detection function """


def get_image_keypoints(mask):
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 255

    params.filterByColor = False
    params.filterByArea = True
    params.minArea = 4
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False

    inv_image = cv2.bitwise_not(mask)
    detector = cv2.SimpleBlobDetector_create(params)

    return detector.detect(inv_image)


""" validate that selected model exist """


def validate_model_exists(model_name):
    if os.path.exists(f"target/{model_name}.pth"):
        return True
    else:
        print("Invalid model file. Valid models: \n")
        valid_models = os.listdir('./target')
        print(valid_models)
        return False


""" get yellow squares dimensions """


def get_yellow_squares_dimension(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([20, 80, 80])   
    upper_bound = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((7,7),np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    try:
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x,y,w,h = cv2.boundingRect(contours[0])
        return w, h
    except:
        w = 0
        h = 0
        return w, h


""" calculate real coords from pixel coords """


def calculate_real_coordinates(x, y, w_real, h_real, w_pixel, h_pixel):
    if h_pixel != 0 and w_pixel != 0:
        x_real = round(x * (w_real/w_pixel), 3)
        y_real = round(y * (h_real/h_pixel), 3)
    else:
        x_real = 0
        y_real = 0

    return x_real, y_real
