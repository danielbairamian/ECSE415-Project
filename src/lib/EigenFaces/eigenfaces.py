from src.helpers import data_loader
from src.helpers import data_preprocessing
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix



# helper that converts input to grayscale
# and flattens it

def flatten_and_gscale(data):
    copydata = copy.deepcopy(data)
    new_data = []
    for d in copydata:
        # don't gscale for now
        temp = d #cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
        temp = temp.flatten()
        new_data.append(temp)

    return np.asarray(new_data)

def normalize_image(image):

    imgcopy = copy.deepcopy(image)
    normalized_img = []


    arraymin = np.amin(imgcopy)
    arraymax = np.amax(imgcopy)

    # normalize between 0 and 1
    for element in imgcopy:
        element = np.divide((element - arraymin), (arraymax - arraymin))
        normalized_img.append(element)

    return np.asarray(normalized_img)

def save_top5_eigenfaces(eigenfaces, meanface, datacropped, shape):
    for i in range (0, 5):
        plt.figure()
        eigenfacetemp = normalize_image(eigenfaces[i])
        eigenfacetemp = eigenfacetemp.reshape(shape, shape, 3)
        plt.imshow(eigenfacetemp)
        plt.xticks([])
        plt.yticks([])
        if datacropped:
            plt.savefig("EigenFacesCropped/EigenFace"+str(i+1)+".png", transparent=True)
        else:
            plt.savefig("EigenFacesFull/EigenFace"+str(i+1)+".png", transparent=True)

        plt.close()
    # also save the mean face
    plt.figure()
    meanfacetemp = normalize_image(meanface)
    meanfacetemp = meanfacetemp.reshape(shape, shape, 3)
    plt.imshow(meanfacetemp)
    plt.xticks([])
    plt.yticks([])
    if datacropped:
        plt.savefig("EigenFacesCropped/MeanFace.png", transparent=True)
    else:
        plt.savefig("EigenFacesFull/MeanFace.png", transparent=True)
    plt.close()


def generate_PCA_Dimensionality_Estimation(data, datacropped, threshold=0.01, save_files = True):

    pca = PCA(n_components=data.shape[0])
    pca.fit(data)

    eigenvalue_eigenvector = list(zip(pca.components_, pca.explained_variance_, pca.explained_variance_ratio_))
    eigenvalue_eigenvector.sort(key=lambda x: x[1], reverse=True)

    idx = 0
    cutoff_threshold = threshold
    for vals in eigenvalue_eigenvector:
        idx += 1
        if vals[2] < cutoff_threshold:
            break

    if save_files:
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_) / np.sum(pca.explained_variance_ratio_), '.-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance')
        if datacropped:
            plt.savefig('VarianceGraphsCropped/Cumulative-Variance.png')
        else:
            plt.savefig('VarianceGraphsFull/Cumulative-Variance.png')

        plt.figure()
        plt.xlabel('Principal Component')
        plt.ylabel('Normalized EigenValue')
        plt.plot(np.linspace(1, data.shape[0], num=data.shape[0]), pca.explained_variance_ratio_)

        plt.plot(np.linspace(1, data.shape[0], num=data.shape[0]),
                 np.linspace(cutoff_threshold, cutoff_threshold, num=data.shape[0]), alpha=0.65, color='orange')
        plt.scatter(idx, cutoff_threshold, color='red')
        plt.text(int(data.shape[0]/2), cutoff_threshold + 0.025,
                 s="PC# " + str(idx) + " | Cuttoff Threshold: " + str(cutoff_threshold * 100) + "%")
        if datacropped:
            plt.savefig('VarianceGraphsCropped/Individual-Variance.png')
        else:
            plt.savefig('VarianceGraphsFull/Individual-Variance.png')
        plt.close()

    return idx


def project_eigenfaces(image_list, labels, eigenvectors, meanface,  metadata, shape, datacropped, save_files = True):
    idx = 1

    train_projected = []

    for datapoint, label in zip(image_list, labels):
        reconstruction = 0
        for eigenF, eigenW in zip(eigenvectors, datapoint):
            reconstruction += eigenF*eigenW

        reconstruction = meanface + reconstruction
        output = copy.deepcopy(reconstruction)
        train_projected.append(output)

        reconstruction = normalize_image(reconstruction)
        reconstruction = reconstruction.reshape(shape, shape, 3)

        if save_files:
            plt.figure()
            plt.imshow(reconstruction)
            plt.xticks([])
            plt.yticks([])
            if datacropped:
                plt.savefig("EigenTrainProjCropped/" + metadata[label] + "_" + str(idx) + ".png", transparent=True)
            else:
                plt.savefig("EigenTrainProjFull/" + metadata[label] + "_" + str(idx) + ".png", transparent=True)

            plt.close()

        idx += 1

    return np.asarray(train_projected)

# classifier function that uses KNN
def classify(X_train, Y_train, X_test):
    # initialize prediction array
    ypred = []

    # for each testpoint
    for testpoint in X_test:

        # initialize a big distance
        min_distance = np.inf
        predclass = -1

        # find the training point with the smallest eucledian distance
        # this is the eucledian distance in eigen space
        # which is much lower in much lower dimension than image space
        for trainpoint, trainlabel in zip(X_train, Y_train):
            dist = np.linalg.norm(testpoint-trainpoint)
            # if we found an image that is closer to our previous guess
            if min_distance > dist:
                # save the label of that image to be our prediction
                min_distance = dist
                predclass = trainlabel
        # append the predicted label to our prediction array
        ypred.append(predclass)
    return ypred


# some metrics to evaluate classifiers
def classifier_evaluator(YTest, YPred):
    # accuracy, confusion matrix (reuqired)
    accuracy = accuracy_score(YTest, YPred)
    confmat = confusion_matrix(YTest, YPred)
    evaluation_metrics = [accuracy, confmat]

    return evaluation_metrics

def evaluation_helper(eval_object, datacropped):
    accuracy = eval_object[0]
    confmat = eval_object[1]

    print("Classifier Accuracy: ", accuracy*100, "%")

    # display
    plt.figure()
    plt.imshow(confmat)
    plt.title(str(accuracy*100) + "% Accuracy: " + " | Confusion Matrix"), plt.xticks([]), plt.yticks([])
    if datacropped:
        plt.savefig("ClassifierResultCropped/ConfusionMatrix.png")
    else:
        plt.savefig("ClassifierResultFull/ConfusionMatrix.png")

    plt.close()


def EigenFaceClassifier(datacropped):
    # load data
    data = data_loader.get_data(isCropped=datacropped)

    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]
    metadata = data[4]

    # X_train and X_test minimum size are the same
    # Ideally want to reshape X_test size with X_train minimum size
    # but here they happen to be identical. Not a problem for our dataset
    # but bad practice.

    # We should be doing what we're doing with the FaceTagging with EigenFaces
    # Which is manually reshape everything to be the same shape of X_train minimum shape
    if datacropped:
        X_train = data_preprocessing.normalize_img_size(X_train)
        X_test = data_preprocessing.normalize_img_size(X_test)

    print(X_train.shape)
    print(X_test.shape)

    imgshape = X_train.shape[1]

    # flatten and convert images to grayscale
    X_train_flat = flatten_and_gscale(X_train)
    X_test_flat = flatten_and_gscale(X_test)

    # get a dimensionality estimation
    optimal_dims = generate_PCA_Dimensionality_Estimation(X_train_flat, datacropped, save_files=True)

    # run PCA with the optimal number of dimensions
    pca = PCA(n_components=optimal_dims)
    # fit our data
    pca.fit(X_train_flat)

    # sort eigenvectors by eigenvalue
    eigenvalue_eigenvector = list(zip(pca.components_, pca.explained_variance_, pca.explained_variance_ratio_))
    eigenvalue_eigenvector.sort(key=lambda x: x[1], reverse=True)

    # separate them
    eigenvectors = np.asarray([x[0] for x in eigenvalue_eigenvector])
    eigenvalues = np.asarray([x[1] for x in eigenvalue_eigenvector])

    # get the mean face
    mean_face = pca.mean_

    # save the top 5 eigen faces
    # only uncomment this if we don't want to save the eigenfaces
    save_top5_eigenfaces(eigenvectors, mean_face, datacropped, imgshape)

    # transform all training images to eigen space
    eigenImagesTrain = pca.transform(X_train_flat)

    # project the train images on the eigen faces to get their representation
    # only use this if we want to save the images again
    train_projected = project_eigenfaces(eigenImagesTrain, Y_train, eigenvectors,
                                         mean_face, metadata, imgshape, datacropped, save_files=True)

    # transform all testing images to eigen space
    eigenImagesTest = pca.transform(X_test_flat)

    # predict test image based on eigenspace eucledian distance
    Y_pred = classify(eigenImagesTrain, Y_train, eigenImagesTest)

    # get the evaluaiton
    evaluation_metrics = classifier_evaluator(Y_test, Y_pred)
    evaluation_helper(evaluation_metrics, datacropped)

if __name__ == "__main__":
    # Run the EigenFace Classifier for both datasets
    EigenFaceClassifier(True)
    EigenFaceClassifier(False)
