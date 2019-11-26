from src.helpers import data_loader
from src.helpers import data_preprocessing
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



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

def save_top5_eigenfaces(eigenfaces):
    for i in range (0, 5):
        plt.figure()
        eigenfacetemp = normalize_image(eigenfaces[i])
        eigenfacetemp = eigenfacetemp.reshape(256, 256, 3)
        plt.imshow(eigenfacetemp)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("EigenFace"+str(i+1)+".png", transparent=True)

def generate_PCA_Dimensionality_Estimation(data, threshold=0.001):

    pca = PCA(n_components=data.shape[0])
    pca.fit(data)

    eigenvalue_eigenvector = list(zip(pca.components_, pca.explained_variance_, pca.explained_variance_ratio_))
    eigenvalue_eigenvector.sort(key=lambda x: x[1], reverse=True)

    idx = 0
    cutoff_threshold = threshold  # 0.1% cuttoff threshold
    for vals in eigenvalue_eigenvector:
        idx += 1
        if vals[2] < cutoff_threshold:
            break

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_) / np.sum(pca.explained_variance_ratio_), '.-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance')
    plt.savefig('Cumulative-Variance.png')

    plt.figure()
    plt.xlabel('Principal Component')
    plt.ylabel('Normalized EigenValue')
    plt.plot(np.linspace(1, data.shape[0], num=data.shape[0]), pca.explained_variance_ratio_)

    plt.plot(np.linspace(1, data.shape[0], num=data.shape[0]),
             np.linspace(cutoff_threshold, cutoff_threshold, num=data.shape[0]), alpha=0.65, color='orange')
    plt.scatter(idx, cutoff_threshold, color='red')
    plt.text(int(data.shape[0]/2), cutoff_threshold + 0.025,
             s="PC# " + str(idx) + " | Cuttoff Threshold: " + str(cutoff_threshold * 100) + "%")
    plt.savefig('Individual-Variance.png')
    plt.close()

    return idx


if __name__ == "__main__":

    # load data
    data = data_loader.get_data(isCropped=False)

    X_train = data[0]
    Y_train = data[1]
    X_test  = data[2]
    Y_test  = data[3]
    metadata= data[4]

    #X_train = data_preprocessing.normalize_img_size(X_train)
    #X_test  = data_preprocessing.normalize_img_size(X_test)

    # flatten and convert images to grayscale
    X_train_flat = flatten_and_gscale(X_train)
    X_test_flat  = flatten_and_gscale(X_test)

    # get a dimensionality estimation
    optimal_dims = generate_PCA_Dimensionality_Estimation(X_train_flat)

    # run PCA with the optimal number of dimensions
    pca = PCA(n_components=optimal_dims)
    # fit our data
    pca.fit(X_train_flat)

    # sort eigenvectors by eigenvalue
    eigenvalue_eigenvector = list(zip(pca.components_, pca.explained_variance_, pca.explained_variance_ratio_))
    eigenvalue_eigenvector.sort(key=lambda x: x[1], reverse=True)

    # separate them
    eigenvectors = np.asarray([x[0] for x in eigenvalue_eigenvector])
    eigenvalues  = np.asarray([x[1] for x in eigenvalue_eigenvector])

    # save the top 5 eigen faces
    save_top5_eigenfaces(eigenvectors)

    # get the mean face
    mean_face = pca.mean_


    # testing eigenface reconstruction with one image
    eigenImages = pca.transform(X_test_flat)
    firstimage = eigenImages[0]
    testimg = X_test_flat[0] - mean_face
    testimg = np.dot(testimg.T, eigenvectors.T)

    reconstruction = 0
    for eigenF, eigenW in zip(eigenvectors, testimg):
        reconstruction += eigenF*eigenW
    reconstruction = mean_face + reconstruction

    reconstruction = normalize_image(reconstruction)
    reconstruction = reconstruction.reshape(256, 256, 3)

    cv2.imshow("init", X_test[0])
    cv2.imshow("reconstruction", reconstruction)
    cv2.waitKey(0)




    #testimg = X_train_flat[0] - mean_face
    #testimg = np.dot(testimg.T, eigenvectors.T)

    #testimg = testimg.reshape(32, 32)
   #plt.imshow(testimg)
   # plt.show()
    #print(testimg.shape)
    #testimg = normalize_image(testimg)
    #plt.imshow(testimg)
    #plt.show()

    # print(np.shape(X_test_flat))
    # print(eigenvectors.shape)
    # print(eigenvalues.shape)
    # print(mean_face.shape)
    # print(eigenvalue_eigenvector[0][0].shape)
    #
    # print(np.shape(pca.mean_))
    # print(np.shape(pca.components_))
    # print(np.shape(pca.explained_variance_))






    # ==================================
    #print(eigenspaceData.shape)
    #print(np.shape(eigenspaceData))
    #print(np.shape(pca.mean_))
    #print(np.shape(pca.components_))
    #print(np.shape(pca.explained_variance_))
    #print(eigenVecs)


    #showface = #eigenVectors[1]
    #showface = normalize_image(showface)
    #print(showface)
    #plt.imshow(showface.reshape(256, 256, 3))
    #print(showface.shape)
    #plt.imshow(showface)
    #plt.show()
    #print(X_train_flat.shape)
    #print(Y_train.shape)
    #
    #print(X_test_flat.shape)
    #print(Y_test.shape)

    # plt.imshow(X_test_flat[0].reshape(256, 256), cmap="gray")
    # plt.show()
    #
    # print(metadata)
    #
    # print(Y_train)
    #
    # plt.imshow(X_train[15])
    # plt.title(metadata[Y_train[15]])
    # plt.show()