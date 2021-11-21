import os

""" Create dir if not exists """
def make_dir(path):
    if os.path.exists(path):
        print("The Specified dir already exists")
    else:
        os.mkdir(path)