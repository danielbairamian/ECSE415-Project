import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


importpath   = "../../Dataset/"
testrainpath = ["Testing", "Training"]
namespath    = ["/Abed/", "/Daniel/", "/Jules/"] # , "/Lea/", "/Patrick/"]
rescaledpath = "Rescaled"
croppedpath  = "Cropped"

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
                faces_detected = face_cascade.detectMultiScale(img_gray, scaleFactor=1.01, minNeighbors=10, minSize=(64, 64))
                for (x, y, w, h) in faces_detected:
                    img_rgb = cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.imwrite(targetfolder+imgname, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    #resize_images()
    crop_images()
