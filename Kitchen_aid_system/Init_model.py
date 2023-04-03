from mrcnn.config import Config
from mrcnn import model as modellib
# from mrcnn import visualize
import mrcnn
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Rectangle
import glob
from skimage.measure import find_contours
import copy

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf

from mrcnn.visualize import display_instances
from mrcnn.visualize import save_instances

model = 0
box_coo_before = 'init'

class myMaskRCNN_Config(Config):
    NAME = "MaskRCNN_inference"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
    # class = 80, add 1 for background


def RCNN_init():
    global model
    config = myMaskRCNN_Config()

    print("loading weights for Mask R-CNN model...")
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
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

"""
def draw_bounding_boxes(filename, boxes_list):
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for box in boxes_list:
        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=5)
        ax.add_patch(rect)
"""

# pyplot.show()


def Make_Mask(frame):
    global model
    # dir_result = os.path.dirname(os.path.abspath(__file__)) + '\\segment_result\\'
    # dir_images = os.path.dirname(os.path.abspath(__file__)) + '\\'  # + '\\segment_test\\'

    # result_list = dir_result + str(i) + '.png'
    # img_read = img_to_array(frame)

    results = model.detect([frame], verbose=0)
    r = results[0]
    # save_instances(img_read, r['rois'], r['masks'], r['class_ids'], class_names, result_list, r['scores'])
    return r
