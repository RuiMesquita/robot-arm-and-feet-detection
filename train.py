from glob import glob
from os import mkdir
from functions import seeding, make_dir, epoch_time


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Create a directory to store the model """
    make_dir("files")

    """ Load dataset """
    train_x = sorted(glob("./augmented_data/train/images/*"))
    train_y = sorted(glob("./augmented_data/train/masks/*"))

    valid_x = sorted(glob("./augmented_data/test/images/*"))
    valid_y = sorted(glob("./augmented_data/test/masks/*"))

    meta_data = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}"
    print(meta_data)