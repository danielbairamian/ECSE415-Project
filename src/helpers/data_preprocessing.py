import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import select

importpath   = "../../Dataset/"
testrainpath = ["Testing", "Training"]
namespath    = ["/Abed/", "/Daniel/", "/Jules/", "/Lea/", "/Patrick/"]
rescaledpath = "Rescaled"
croppedpath  = "BoxCropped"

ViolaJonesXML = "haarcascade_frontalface_default.xml"

resizedim = (256, 256)


face_cascade = cv2.CascadeClassifier(importpath+ViolaJonesXML)

def resize_images():
    # get all folders
    for ttpath in testrainpath:
        # get all names
        for npath in namespath:

            # set the source folder
            imagefolder  = importpath + ttpath                + npath
            # set the target folder
            targetfolder = importpath + ttpath + rescaledpath + npath

            # get all images from the folder
            for imgname in os.listdir(imagefolder):
                # read, resize, write
                img = cv2.imread(imagefolder+imgname)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img , resizedim)

                cv2.imwrite(targetfolder+imgname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def crop_image_helper(x, y, w, h, img_rgb):
    img_rgb_clean = copy.deepcopy(img_rgb)
    img_rgb = cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # break after detecting 1 face, we'll make sure it's the optimal one
    while (True):
        temp = cv2.cvtColor(copy.deepcopy(img_rgb), cv2.COLOR_BGR2RGB)
        cv2.imshow('image', temp)
        # control movement of box
        user_input = cv2.waitKey(0)
        if user_input == ord('q'):
            img_rgb = img_rgb_clean[y:y + h, x:x + w]
            break
        elif user_input == ord('w'):
            y -= 8
        elif user_input == ord('a'):
            x -= 8
        elif user_input == ord('s'):
            y += 8
        elif user_input == ord('d'):
            x += 8

        # control size of box
        elif user_input == ord('i'):
            w += 4
            h += 4

        elif user_input == ord('o'):
            w -= 4
            h -= 4


        img_rgb = cv2.rectangle(copy.deepcopy(img_rgb_clean), (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img_rgb

def crop_images():

    # get all folders
    for ttpath in testrainpath:
        # get all names
        for npath in namespath:

            # set the source folder
            imagefolder  = importpath + ttpath + rescaledpath + npath
            # set the target folder
            targetfolder = importpath + ttpath + croppedpath  + npath

            # get all images from the folder
            for imgname in os.listdir(imagefolder):
                img      = cv2.imread(imagefolder + imgname)
                img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Need to tweak these values
                faces_detected = face_cascade.detectMultiScale(img_gray, scaleFactor=1.01, minNeighbors=10,minSize=(64, 64))

                # if we didn't find any faces
                if len(faces_detected) == 0:
                    x = 128
                    y = 128
                    w = 64
                    h = 64

                    img_rgb = crop_image_helper(x, y, w, h, img_rgb)
                # if we did find at least one face
                else:
                    for (x, y, w, h) in faces_detected:
                        img_rgb = crop_image_helper(x, y, w, h, img_rgb)
                        # we know we only have 1 face, break
                        break
                cv2.imwrite(targetfolder+imgname, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def crop_images_no_box():

    # get all folders
    for ttpath in testrainpath:
        # get all names
        for npath in namespath:

            # set the source folder
            imagefolder  = importpath + ttpath + rescaledpath + npath
            # set the target folder
            targetfolder = importpath + ttpath + croppedpath  + npath

            # get all images from the folder
            for imgname in os.listdir(imagefolder):
                img      = cv2.imread(imagefolder + imgname)
                img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Need to tweak these values
                faces_detected = face_cascade.detectMultiScale(img_gray, scaleFactor=1.01, minNeighbors=10,minSize=(64, 64))

                # if we didn't find any faces
                if len(faces_detected) == 0:
                    x = 128
                    y = 128
                    w = 64
                    h = 64

                    img_rgb = crop_image_helper(x, y, w, h, img_rgb)
                # if we did find at least one face
                else:
                    for (x, y, w, h) in faces_detected:
                        img_rgb = crop_image_helper(x, y, w, h, img_rgb)
                        # we know we only have 1 face, break
                        break
                cv2.imwrite(targetfolder+imgname, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


# helper to make all cropped images the same size
# this is so algo's that require images to be the same size to work (PCA for example)
def normalize_img_size(imglist):

    list_of_uneven_imgs = copy.deepcopy(imglist)
    samesize_imgs = []

    min_shape = np.inf
    # get the smallest size image
    for unevenimg in list_of_uneven_imgs:
        if unevenimg.shape[0] < min_shape:
            min_shape = unevenimg.shape[0]

    # save the smallest dimension
    smallestdim = (min_shape, min_shape)

    # resize all images to the smallest size
    for unevenimg in list_of_uneven_imgs:
        unevenimg = cv2.resize(unevenimg, smallestdim)
        samesize_imgs.append(unevenimg)

    return np.asarray(samesize_imgs)


if __name__ == "__main__":
    #resize_images()
    #crop_images()
    crop_images_no_box()
