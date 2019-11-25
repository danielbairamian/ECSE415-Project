from src.helpers import data_loader
from src.helpers import hog

import matplotlib.pyplot as plt

import cv2

if __name__ == '__main__':

    # Read in data
    data = data_loader.get_data(isCropped=True)
    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]
    metadata = data[4]

    # Extract patches of three different sizes around

    plt.figure(figsize=(5,5))
    plt.imshow(X_train[0])
    plt.show()


    # Detect HoG features for each image