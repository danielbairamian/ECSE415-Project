from src.helpers import data_loader
from src.helpers import data_preprocessing
from src.lib.EigenFaces import eigenfaces

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from src.helpers.TemplateMatching import get_coordinates

from src.lib.EigenFaces import eigenfaces


def tag_group_pictures(group_photo, cropped_photos, labels):
    # save all coordinates
    coords_list = []
    for cp in cropped_photos:
        coords = get_coordinates(group_photo, cp)
        coords = [coords[0], coords[1], cp.shape[0], cp.shape[1]]
        coords_list.append(coords)

    # make a copy of our group photo
    temp_img_final = copy.deepcopy(group_photo)
    # for each coordinate we have
    # place the bounding box and tag the image
    for i, coord in enumerate(coords_list):
        x = coord[0]
        y = coord[1]

        w = coord[2]
        h = coord[3]

        # bounding box variables
        start_point = (x, y)
        end_point   = (x+w, y+h)

        color = (255, 0, 0)
        thickness = 1

        # text variables
        font = cv2.FONT_HERSHEY_PLAIN
        bottomLeftCornerOfText = start_point
        fontScale = ((w+h))*3 / group_photo.shape[0]
        fontColor = (0, 255, 0)
        lineType = 1

        # put the text and the bounding box on the image
        tempImg = cv2.rectangle(group_photo, start_point, end_point, color, thickness)
        cv2.putText(tempImg, labels[i], bottomLeftCornerOfText, font,fontScale, fontColor, lineType)

        temp_img_final = copy.deepcopy(tempImg)

    # return the final tagged image
    return temp_img_final

# For face tagging, we know each image has each of our 5 training subjects
# Instead of using regular KNN, we can use this information to limit the classifier
# to only predict each label one time

# To do so, we iteratively go through our test points,
# and each time we save the label with lowest overall prediction error
# this is the "confidence" of the classification
# and then discard all other classifications

# we then mark the label and the test point that got classified, then rerun again
# repeat until we don't have unclassified labels

def Iterative_Confidence_Classifier(X_train, Y_train, X_test):
    # initialize prediction array

    # create a placeholder prediction list
    final_prediction = np.ones(X_test.shape[0])*(-1)
    #array that holds the labels that have been used already
    already_predicted_labels = []
    # array that holds the indices of the test points that have been classified
    already_predicted_indices= []

    # for every datapoint
    for i in range (0, len(X_test)):
        #print("=================")
        # initialize our prediction tentative
        ypred = []
        # for all datapoints
        for k in range(0, len(X_test)):
            # skip the ones already predicted
            if k in already_predicted_indices:
                continue
            # get the test point
            tp = X_test[k]
            min_distance = np.inf
            predclass = -1

            # for all training points
            for trainpoint, trainlabel in zip(X_train, Y_train):
                # if the training point holds one of the labels we already used
                # then we can't use that training point, skip it
                if trainlabel in already_predicted_labels:
                    continue
                # else, try to classify
                dist = np.linalg.norm(tp - trainpoint)
                if min_distance > dist:
                    min_distance = dist
                    predclass = trainlabel
            # save the prediction, the distance, and the index
            ypred.append([predclass, min_distance, k])

        # after all training points have been classified
        min_label = -1
        min_val = np.inf
        min_idx = -1

        # find the training point that got classified with the least error
        # this classification is our most confident guess
        for pred in ypred:
            #print(pred)
            if pred[1] < min_val:
                min_label = pred[0]
                min_val   = pred[1]
                min_idx   = pred[2]

        # save the label of most confident guess
        already_predicted_labels.append(min_label)

        # save the index of the testpoint with this classification
        already_predicted_indices.append(min_idx)

        # save the label to our final prediction list at the correct position
        final_prediction[min_idx] = min_label

    return final_prediction

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
    # tweaked a bit n components
    pca = PCA(n_components=12)
    # fit our data
    pca.fit(X_train_flat)

    # transform all training images to eigen space
    eigenImagesTrain = pca.transform(X_train_flat)


    final_tagged_images = []

    # for all groups that we have
    for group in All_Groups:
        # the first image is the group picture
        original_group = group[0]
        cropped_group  = []
        cropped_group_resized = []
        # we have 5 trained individuals, so for 1->5 (0 is the group)
        for i in range (1, 6):
            # save the image
            temp_img = group[i]
            # save original sized image
            cropped_group.append(copy.deepcopy(temp_img))
            # save a resized image to fit to PCA
            cropped_group_resized.append(cv2.resize(copy.deepcopy(temp_img), (imgshape, imgshape)))

        # flatten and project resized faces to eigenspace
        cropped_group_resized = np.asarray(cropped_group_resized)
        cropped_group_resized = eigenfaces.flatten_and_gscale(cropped_group_resized)
        cropped_group_resized = pca.transform(cropped_group_resized)

        # run our classifier to get prediction labels
        ypred_temp = Iterative_Confidence_Classifier(eigenImagesTrain, Y_train, cropped_group_resized)
        #ypred_temp = eigenfaces.classify(eigenImagesTrain, Y_train, cropped_group_resized)

        # get the name associated to the labels
        labels_tags = [metadata[x] for x in ypred_temp]

        # helper function that will place the name and the box around the group image
        final_tagged_images.append(tag_group_pictures(original_group, cropped_group, labels_tags))

    for i, taggedIm in enumerate(final_tagged_images):
        plt.figure()
        plt.imshow(taggedIm)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("GroupImgs/Group"+str(i+1)+".png", transparent=True)
        plt.close()


if __name__ == "__main__":
    FaceTagging(True)