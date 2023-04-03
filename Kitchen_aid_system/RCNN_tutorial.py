from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import glob
import time

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from mrcnn.visualize import display_instances
from mrcnn.visualize import save_instances
import tensorflow as tf


class myMaskRCNN_Config(Config):
    NAME = "MaskRCNN_inference"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
    # class = 80, add 1 for background


config = myMaskRCNN_Config()

print("loading weights for Mask R-CNN model...")
model = modellib.MaskRCNN(mode = "inference", config = config, model_dir = './')
model.load_weights('mask_rcnn_coco.h5', by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
 'bus', 'train', 'truck', 'boat', 'traffic light',
 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
 'kite', 'baseball bat', 'baseball glove', 'skateboard',
 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
 'teddy bear', 'hair drier', 'toothbrush']


def draw_bounding_boxes(filename, boxes_list):
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for box in boxes_list:
        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'red', lw = 5)
        ax.add_patch(rect)

 #pyplot.show()


dir_result = os.path.dirname(os.path.abspath(__file__)) + '\\segment_result\\'
dir_images = os.path.dirname(os.path.abspath(__file__))+ '\\images\\' # + '\\segment_test\\'
i = 0


img_list = glob.glob(dir_images + '*.jpg')
"""
img1 = load_img('segtest_1.jpg')
img2 = load_img('segtest_2.jpg')
# pyplot.imshow(img)
img1 = img_to_array(img1)
img2 = img_to_array(img2)
results = model.detect([img1, img2], verbose = 0)
# draw_bounding_boxes('donuts.jpg', results[0]['rois'])

r = results[0]
print(r)
save_instances(img1, r['rois'], r['masks'], r['class_ids'], class_names, '1.jpg', r['scores'])
save_instances(img2, r['rois'], r['masks'], r['class_ids'], class_names, '2.jpg', r['scores'])
"""
for img in img_list:
    result_list = dir_result + str(i) + '.png'
    print(img)
    startTime = time.time()

    img_read = load_img(img)
    img_read = img_to_array(img_read)
    # img_part1, img_part2, img_part3, img_part4 = np.split(img_read, 4, axis = 0)
    # np.split(img_read, 50, axis=0)

    # print(img_part2.shape)
    print(img_read.shape)
    results = model.detect([img_read], verbose = 0)
    r = results[0]
    save_instances(img_read, r['rois'], r['masks'], r['class_ids'], class_names, result_list, r['scores'])
    # display_instances(img_read, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    i += 1
    endTime = time.time() - startTime
    print(round(endTime, 2), 'Second')
    break

# (448, 640, 3) / rows, cols, dimensions
