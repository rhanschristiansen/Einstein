import cv2
import numpy as np

def min_max_hsv_from_bb(hsv_img, bbox):
    hsv_img_crop = hsv_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    h_ch = hsv_img_crop[:,:,0]
    s_ch = hsv_img_crop[:,:,1]
    v_ch = hsv_img_crop[:,:,2]

    h_min = np.min(h_ch)
    s_min = np.min(s_ch)
    v_min = np.min(v_ch)
    h_max = np.max(h_ch)
    s_max = np.max(s_ch)
    v_max = np.max(v_ch)

    return [h_min, s_min, v_min, h_max, s_max, v_max]


def on_mouse(event, x, y, flags, param):
    img, draw_img, img_hsv, img_binary, hsv_filter, bbox, dragging = param
    draw_img = img.copy() # refresh drawing image
    pt = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN and dragging is False:
        bbox[0] = x
        bbox[1] = y
        bbox[2] = x
        bbox[3] = y
        dragging = True

    if event == cv2.EVENT_LBUTTONDOWN and dragging is True:
        bbox[2] = x
        bbox[3] = y

    if event == cv2.EVENT_LBUTTONUP:
        dragging = False
        bbox[2] = x
        bbox[3] = y

        # bounding box has been selected, now find min/max hsv values from this
        hsv_filter = min_max_hsv_from_bb(img_hsv, bbox)
        lowerb = np.array(hsv_filter[:3])
        upperb = np.array(hsv_filter[3:])
        img_binary = cv2.inRange(img_hsv, lowerb,upperb)
        print hsv_filter


    cv2.rectangle(draw_img, tuple(bbox[:2]), tuple(bbox[2:]),(0,255,0),2)

# img = cv2.imread('/home/robert/PycharmProjects/sandbox/tracking/balls1.png')
img = cv2.imread('/home/robert/Pictures/balls2.png')
video_filename = '/home/robert/datasets/video/TrackBalls/2017-05-12-112517.webm'
vc = cv2.VideoCapture()
vc.open(video_filename)
vc.set(cv2.CAP_PROP_POS_FRAMES, 100)
_, img = vc.read()
draw_img = img.copy()
cv2.namedWindow('draw_img')
dragging = False
# convert to hsv
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_filter = [0, 0, 0, 255, 255, 255]
bbox = [0,0,0,0]
img_binary = cv2.inRange(img_hsv, tuple(hsv_filter[:3]), tuple(hsv_filter[3:]))
param = img, draw_img, img_hsv, img_binary, hsv_filter, bbox, dragging
cv2.setMouseCallback('draw_img', on_mouse,param=param)

while True:
    cv2.imshow('draw_img', draw_img)
    cv2.imshow('img_hsv', img_hsv)
    # cv2.imshow('img_binary', img_binary)
    cv2.waitKey(30)