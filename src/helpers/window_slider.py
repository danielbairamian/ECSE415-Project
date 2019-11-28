import numpy as np
def sliding_window(image, cropped_img, x_stride=1, y_stride=1):
    windows = []
    windows_pos = []
    winsize_x = cropped_img.shape[1]
    winsize_y = cropped_img.shape[0]

    y_range = image.shape[0]
    x_range = image.shape[1]
    min_distance = np.inf
    for i in range(0, y_range, y_stride):
        for j in range(0, x_range, x_stride):
            local_window = image[i:i + winsize_y, j:j + winsize_x]
            #             if local_window.shape[0] != cropped_img.shape[0] or local_window.shape[1] != cropped_img.shape[1]:
            #                 continue
            #             else:
            cropped_img = np.resize(125, 125)
            local_window = np.resize(125, 125)

            dist = np.linalg.norm(cropped_img - local_window)
            # if we found an image that is closer to our previous guess
            if min_distance > dist:
                # save x and y
                min_distance = dist
                x = j
                y = i
    return x, y
