import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


""" Create dir if not exists """
def make_dir(path):
    if os.path.exists(path):
        print("The Specified dir already exists")
    else:
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
    file = open("trainning_report.txt", "a")
    file.write(epoch_status)
    file.write("\n")
    file.close()


""" Store metrcis report for each test """
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
    
    number_of_dirs += 1
    return f"report{number_of_dirs:03}"
