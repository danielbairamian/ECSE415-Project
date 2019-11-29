from src.helpers import data_loader
from src.helpers import patch_maker
from src.helpers import hog
from src.lib.SIFT import sift_sizes
from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans



import matplotlib as mpl
import numpy as np

def patches_and_sift(imgs, imgs_cropped, patch_size, radius):
    descriptors = []
    descriptors_not_norm = []
    for i in range(len(imgs)):
        k = sift_sizes.sift(imgs_cropped[i])
        patch = patch_maker.make_patch(imgs[i], imgs_cropped[i], patch_size, k[:15])
        descr = hog.hog_features(patch, radius)
        descriptors_not_norm.append(descr)
        descriptors.append(np.asarray(descr).flatten().ravel())
    return descriptors, descriptors_not_norm

def bag_of_words():
    return 0



if __name__ == '__main__':
    # Read in data
    cropped_data = data_loader.get_data(isCropped=True, doshuffle=False)
    X_train_cropped = cropped_data[0]
    Y_train_cropped = cropped_data[1]
    X_test_croppped = cropped_data[2]
    Y_test_cropped = cropped_data[3]
    metadata_cropped = cropped_data[4]

    data = data_loader.get_data(isCropped=False, doshuffle=False)
    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]
    metadata = data[4]

    patches = [6, 10, 15, 20]
    best_cell_size = 4
    try:
        descriptors_train, descriptors_not_norm = patches_and_sift(X_train, X_train_cropped, patches[3], best_cell_size)
        descriptors_train = np.asarray(descriptors_train)

        min_l = len(descriptors_not_norm[0])
        hog_l = len(descriptors_not_norm[0][0])

        tests = []
        for i in range(75):
            tests.append(np.asarray(descriptors_not_norm[i][:min_l]))

        print(np.asarray(tests).shape)

        resh = np.reshape(tests, (15 * 75, len(descriptors_not_norm[0][0])))
        print(resh.shape)

        gmm = GaussianMixture(covariance_type='full', max_iter=5)
        print('gmm start')
        gmm.fit(resh)
        print('gmm fit')

        '''
        
        km = KMeans(n_clusters=50)
        km_fit = km.fit(descriptors_not_norm)
        clusters = km_fit.cluster_centers_

        # Compute normalized histogram of words for each training image using Bag of Words (BoW) method.

        # features = 15,  words = 50,   images = 75

        # for each image
            # Iterate through all the features
                # find closest word for that features
        n_words = 15
        n_clusters = 50

        bag_of_words = []
        for i in range(len(X_train)):
            feat_to_word = []
            for j in range(n_words):
                descriptor_keypoint = descriptors_not_norm[i][j]
                min_distance = 2e10
                min_word_indx = 0
                for c in range(n_clusters):
                    cluster_center = clusters[c]
                    dist = np.linalg.norm(descriptor_keypoint - cluster_center)
                    if (min_distance < dist):
                        min_distance = dist
                        min_word_indx = j
                feat_to_word.append(min_word_indx)
            bag_of_words.append(feat_to_word)

        print(bag_of_words)
        '''

        '''
        bag_of_words = []
        for i in range(len(X_train)):
            feat_to_word = []
            for j in range(n_words):
                descriptor_keypoint = descriptors_train[i][j]
                min_distance = 2e10
                min_word_indx = 0
                for c in range(n_clusters):
                    cluster_center = clusters[c]
                    dist = np.linalg.norm(descriptor_keypoint - cluster_center)
                    if(min_distance < dist):
                        min_distance = dist
                        min_word_indx = j
                feat_to_word.append(min_word_indx)
            bag_of_words.append(feat_to_word)
      
        print(bag_of_words)

        # count occurence of bag of words


        # Store the computed histograms. Display the histograms for 3 selected images.


        descriptors_test = patches_and_sift(X_test, X_test_croppped, patches[3], best_cell_size)
        descriptors_test = np.asarray(descriptors_train)

        a = km.predict(descriptors_test)
        '''

        '''
        correct = 0
        wrong = 0
        for i in range(len(a)):

            if metadata[a[i]] == metadata[i%5]:
                correct+=1
            else:
                wrong+=1

        print(correct)
        print(wrong)
        
        #gmm.fit(np.asarray(descriptors))

        
        print(np.shape(descriptors))
        print(descriptors.shape)

        print(type(descriptors))
        print(type(descriptors[0]))

        print(len(descriptors[0]))
        print(len(descriptors[1]))
        print(len(descriptors[2]))
        '''

        # Cluster feature descriptors into K clusters using Gaussian Mixture Model.
        # Each cluster center represents one word.


    except Exception as e:
        print('ERROR:')
        print(e)

