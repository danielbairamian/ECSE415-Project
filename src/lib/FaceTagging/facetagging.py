from src.helpers import data_loader
from src.helpers import data_preprocessing
from src.lib.EigenFaces import eigenfaces

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix


from src.lib.EigenFaces import eigenfaces

def FaceTagging(datacropped=True):
    All_Groups = data_loader.get_group_data()

    # load data
    data = data_loader.get_data(isCropped=datacropped)

    X_train = data[0]
    Y_train = data[1]
    metadata = data[4]

    # resize train and test
    if datacropped:
        X_train = data_preprocessing.normalize_img_size(X_train)

    # shape of X_train minimum
    imgshape = X_train.shape[1]

    # flatten and convert images to grayscale
    X_train_flat = eigenfaces.flatten_and_gscale(X_train)

    # run PCA with the optimal number of dimensions
    # We know this #14 is optimal from part 5
    pca = PCA(n_components=14)
    # fit our data
    pca.fit(X_train_flat)

    # transform all training images to eigen space
    eigenImagesTrain = pca.transform(X_train_flat)

    # load our group data
    testbatch = []
    for i in range (1, 6):
        imgtemp = All_Groups[0][i]
        imgtemp = cv2.resize(imgtemp, (imgshape, imgshape))
        testbatch.append(imgtemp)
        print(np.shape(imgtemp))
    print(np.shape(testbatch))
    testbatch = np.asarray(testbatch)
    testbatch = eigenfaces.flatten_and_gscale(testbatch)

    eigentest = pca.transform(testbatch)
    ypred = eigenfaces.classify(eigenImagesTrain, Y_train, eigentest)
    print(ypred)

if __name__ == "__main__":
    FaceTagging(True)