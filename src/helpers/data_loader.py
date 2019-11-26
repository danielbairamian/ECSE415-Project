import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import select

from sklearn.utils import shuffle


importpath   = "../../../Dataset/"
testrainpath = ["Testing", "Training"]
namespath    = ["/Abed/", "/Daniel/", "/Jules/", "/Lea/", "/Patrick/"]
rescaledpath = "Rescaled"
croppedpath  = "Cropped"

meta_data = {0: 'Abed', 1: 'Daniel', 2: 'Jules', 3: 'Lea', 4: 'Patrick'}

def get_data(isCropped=True, doshuffle = True):

    # Test train data
    X_train = []
    Y_train = []

    X_test  = []
    Y_test  = []

    # define both import paths
    testpath = importpath+testrainpath[0]
    trainpath= importpath+testrainpath[1]

    for i, npath in enumerate(namespath):

        if isCropped:
            testfolder = testpath + croppedpath + npath
            trainfolder= trainpath+ croppedpath + npath
        else:
            testfolder = testpath + rescaledpath + npath
            trainfolder= trainpath+ rescaledpath + npath

        for imgname in os.listdir(testfolder):

            img = cv2.imread(testfolder+imgname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            X_test.append(img)
            Y_test.append(i)

        for imgname2 in os.listdir(trainfolder):

            img = cv2.imread(trainfolder+imgname2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            X_train.append(img)
            Y_train.append(i)

    if doshuffle:
        # once the data is saved, shuffle it
        X_train , Y_train = shuffle(X_train, Y_train)
        X_test  , Y_test  = shuffle(X_test, Y_test)

    return np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), meta_data


def get_all_data():
    data_full    = get_data(isCropped=False, doshuffle=False)
    data_cropped = get_data(isCropped=True,  doshuffle=False)

    X_train = data_cropped[0]
    Y_train = data_cropped[1]

    X_test = data_cropped[2]
    Y_test = data_cropped[3]

    metadata = data_cropped[4]

    X_train_Full = data_full[0]
    X_test_Full = data_full[2]

    # once the data is saved, shuffle it
    X_train, X_train_Full,  Y_train = shuffle(X_train, X_train_Full, Y_train)
    X_test, X_test_Full, Y_test = shuffle(X_test, X_test_Full,  Y_test)

    return X_train, Y_train, X_test, Y_test, metadata, X_train_Full, X_test_Full



