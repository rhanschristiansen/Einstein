"""
A collection of useful image operations for Computer Vision
"""
import cv2
import numpy as np


def load_and_show_img(filename):
    img = cv2.imread(filename=filename)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def get_car_dist_from_image(image):
    print 'getting car distance from image...'
    return 47


def draw_bbox(img, bbox, color=(0, 255, 0)):
    # top left of bbox
    top_left = (bbox[0], bbox[1])
    # bottom right of bbox
    bottom_right = (bbox[2], bbox[3])
    cv2.rectangle(img, top_left, bottom_right, color, 2)


# this function will pad the image to a certain aspect ratio before resizing it
# this is good because it does not distort the image
def resize_pad_image(img, new_dims, pad_output=True):
    old_height, old_width, ch = img.shape
    old_ar = float(old_width) / float(old_height)
    new_ar = float(new_dims[0]) / float(new_dims[1])
    undistorted_scale_factor = [1.0, 1.0]  # if you want to resize bounding boxes on a padded img you'll need this
    if pad_output is True:
        if new_ar > old_ar:
            new_width = old_height * new_ar
            padding = abs(new_width - old_width)
            img = cv2.copyMakeBorder(img, 0, 0, 0, int(padding), cv2.BORDER_CONSTANT, None, [0, 0, 0])
            undistorted_scale_factor = [float(old_width) / (float(new_dims[1]) * old_ar),
                                        float(old_height) / float(new_dims[1])]
        elif new_ar < old_ar:
            new_height = old_width / new_ar
            padding = abs(new_height - old_height)
            img = cv2.copyMakeBorder(img, 0, int(padding), 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])
            undistorted_scale_factor = [float(old_width) / float(new_dims[0]),
                                        float(old_height) / (float(new_dims[0]) / old_ar)]
        elif new_ar == old_ar:
            scale_factor = float(old_width) / new_dims[0]
            undistorted_scale_factor = [scale_factor, scale_factor]
    outimg = cv2.resize(img, (new_dims[0], new_dims[1]))
    return outimg, undistorted_scale_factor
