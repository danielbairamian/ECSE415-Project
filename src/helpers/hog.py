from skimage import feature as ft
import cv2

def hog_ft(imgs, bins=9, blocks=(8,8), cells=(2,2)):
    '''
    Returns hog feats array
    '''
    hog_feats = []
    for i in range(len(imgs)):
        image = imgs[i]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x = ft.hog(gray, orientations=bins, pixels_per_cell=blocks,
                   cells_per_block=cells, block_norm='L2-Hys')
        hog_feats.append(x)
    return hog_feats