from src.helpers import data_loader
from src.helpers import hog
from src.helpers import patch_maker
import matplotlib.pyplot as plt

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
    cropped_data = data_loader.get_data(isCropped=True)
    X_train_cropped = cropped_data[0]
    Y_train_cropped = cropped_data[1]
    X_test_croppped = cropped_data[2]
    Y_test_cropped = cropped_data[3]
    metadata_cropped = cropped_data[4]

    data = data_loader.get_data(isCropped=False)
    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]
    metadata = data[4]

    # Display Cropped and Original Image
    plt.imshow(X_train_cropped[0])
    plt.show()
    plt.imshow(X_train[0])
    plt.show()

    cropped_img = X_train_cropped[0]
    original_img = X_train[0]

    # Get Sift Keypoints coordinates on cropped image
    kp = sift(cropped_img)

    # Patch image
    patches = patch_maker.make_patch(img=original_img, img_cropped=cropped_img, size=15, k=kp)
