from src.helpers import data_loader
import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    data = data_loader.get_data(isCropped=False)

    X_train = data[0]
    Y_train = data[1]
    X_test  = data[2]
    Y_test  = data[3]
    metadata= data[4]

    print(X_train.shape)
    print(Y_train.shape)

    print(X_test.shape)
    print(Y_test.shape)

    print(metadata)

    print(Y_train)

    plt.imshow(X_test[15])
    plt.title(metadata[Y_test[15]])
    plt.show()