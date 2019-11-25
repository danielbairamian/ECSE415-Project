from src.helpers import data_loader
from src.helpers import hog

import matplotlib.pyplot as plt

import cv2

if __name__ == '__main__':

    # Read in data
    data = data_loader.get_data(isCropped=False)
    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]
    metadata = data[4]

    # Get HoG feats for x_train images
    X_hog = hog.hog_ft(X_train)

    sift = cv2.xfeatures2d.SIFT_create()


    # Detect HoG features for each image