from src.helpers import data_loader
from src.helpers import hog
from src.helpers import patch_maker
import matplotlib.pyplot as plt
from src.helpers import hog

import cv2

def sift(img):
    # copy image
    img_disp = img.copy()
    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)

    # create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    # detect SIFT features, with no masks
    keypoints = sift.detect(img, None)

    p = []
    keypoints3, descriptors = sift.compute(img, keypoints)
    for k in keypoints3:
        p.append(k.pt)

    # draw the keypoints
    cv2.drawKeypoints(img, keypoints, img_disp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #     display
    plt.figure(figsize=(10, 10))
    plt.subplot(121), plt.imshow(img)
    plt.title("Input Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_disp)
    plt.title("SIFT Features"), plt.xticks([]), plt.yticks([])
    plt.show()

    # num of SIFT keypoints
    print('Num keypoints: ' + str(len(keypoints)))

    return p

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

    # Display Cropped and Original Image
    plt.imshow(X_train_cropped[1])
    plt.show()
    plt.imshow(X_train[1])
    plt.show()

    cropped_img = X_train_cropped[1]
    original_img = X_train[1]

    # Get Sift Keypoints coordinates on cropped image
    kp = sift(cropped_img)

    # Patch image
    patches = patch_maker.make_patch(img=original_img, img_cropped=cropped_img, size=15, k=kp)

    '''
    Extract HoG descriptor for the patch as follows.
    Fix the block size= 2, number of bins = 9 and vary the cell size. 
    Use cell sizes 3 × 3, 4 × 4 and 5 × 5 . 
    Build separate vocabularies for HoG descriptor with different cell sizes. 
    Test and evaluate the vocabularies using recognition rate. 
    Plot recognition rate (on y-axis) vs cell size of the HoG descriptor (on x-axis). 
    Compute confusion matrix for the best performing vocabulary.
    '''
    hog_3 = hog.hog_features(patches, 3)
    hog_4 = hog.hog_features(patches, 4)
    hog_5 = hog.hog_features(patches, 5)
