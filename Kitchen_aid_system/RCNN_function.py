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
"""
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
"""

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
"""

def apply_mask(image, mask, color, alpha=0.5):
    # Just apply 1 masks in picture. variable padded mask is what I found.
    """Apply the given mask to the image. In visualize file.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def make_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

    return colors


def make_masked_image(image, boxes, masks, class_ids, cut_len,  # class_names, # dir,
                      scores=None,  title="",
                      figsize=(16, 16), ax=None,
                      show_mask=False, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    global class_names
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or make_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint8).copy()
    # bbox_image = image.astype
    padded_mask = 0
    contours = 0
    boxed_image = 0
    only_box = 0
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)  # apply_mask(image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.

        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask

        # contours = find_contours(padded_mask, 0.5)
        # print("padded mask: ", np.shape(padded_mask))
        contours, _ = cv2.findContours(padded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

        # contours_array = np.asarray(contours)
        # print(np.shape(contours_array))
        # print(type(contours_array))
        # contours_array = np.squeeze(contours_array)
        # print("contoures_array: ", contours_array)
        # print(type(contours_array))
        # np.savetxt('contours_array.txt', contours_array)

        # left_top = np.array([height, width])
        # right_bot = np.array([0, 0])

        # for contour in contours_array:
            # contour = contour.flatten()
            # print(contour)
            # np.savetxt('contour.txt', contour)
            # print(left_top, contour[0], 'left top & contour')
            # print(right_bot, 'right bot')
        box_coo = cv2.minAreaRect(contours[0])
        """
            if left_top[0] > contour[0]:
                left_top[0] = contour[0]

            if left_top[1] > contour[1]:
                left_top[1] = contour[1]

            if right_bot[0] < contour[0]:
                right_bot[0] = contour[0]

            if right_bot[1] < contour[1]:
                right_bot[1] = contour[1]

        right_bot = [right_bot[0], right_bot[1]]
        left_top = [left_top[0], left_top[1]]
        """
        # print(box)
        # boxed_image = draw_box(masked_image, left_top, right_bot, color)
        # print(type(color))
        background_matrix = np.full_like(masked_image, 255)
        class_id = class_ids[i]
        print(class_names[class_id])

        # Filtering label and box size
        if class_names[class_id] is 'toothbrush' :
            print('Detected', class_names[class_id], class_id)
            color = [0, 0, 1.0]
            boxed_image = draw_box(masked_image, box_coo, cut_len, color)
            only_box = draw_box(background_matrix, box_coo, cut_len, color)

        # print('no carrot detected')

        # boxed_image = draw_box(masked_image, box_coo, cut_len, color)
        # only_box = draw_box(background_matrix, box_coo, cut_len, color)

    return boxed_image, only_box


def draw_box(image, box_coo, length, color):
    """
    Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """

    color = np.array(color)
    # print(type(color), "color type")
    # print(color, "color")
    color = color * 255
    color[0] = int(color[0])
    color[1] = int(color[1])
    color[2] = int(color[2])

    """
    Rotated bounding box with guide line
    """
    global box_coo_before
    long_width = 0
    img_box = 0

    try:
        if box_coo_before == 'init':
            box_coo_before = copy.deepcopy(box_coo)
            print("box_coo_before initialize \n")

        coordinate_before = np.asarray(box_coo_before[0])
        coordinate_now = np.asarray(box_coo[0])
        angle_before = box_coo_before[2]
        angle_now = box_coo[2]
        angle_difference = abs(angle_now) - abs(angle_before)
        # print(angle_before)
        # print("before", box_coo_before)
        # print("now", box_coo)

        # Angle margin
        """
        if angle_now > 0:
            if angle_before * 0.8 < angle_now < angle_before * 1.2:
                rect_for_line = box_coo_before
                box_coo = cv2.boxPoints(box_coo_before)
                box_coo = box_coo.astype('int')
                print("In margin")

            else:
                box_coo_before = copy.deepcopy(box_coo)
                rect_for_line = box_coo
                box_coo = cv2.boxPoints(box_coo)
                box_coo = box_coo.astype('int')

        else:
            if angle_before * 0.8 > angle_now > angle_before * 1.2:
                rect_for_line = box_coo_before
                box_coo = cv2.boxPoints(box_coo_before)
                box_coo = box_coo.astype('int')
                print("In margin")

            else:
                box_coo_before = copy.deepcopy(box_coo)
                rect_for_line = box_coo
                box_coo = cv2.boxPoints(box_coo)
                box_coo = box_coo.astype('int')
        """
        if abs(angle_difference) < 10:
            # If Coordinate is change but angle does not, It would be helpful.
            if cv2.norm(coordinate_before - coordinate_now, cv2.NORM_L2) > 30:
                box_coo_before = copy.deepcopy(box_coo)
                rect_for_line = box_coo
                box_coo = cv2.boxPoints(box_coo)
                box_coo = box_coo.astype('int')

            else:
                # print("angle difference: ", angle_difference)
                rect_for_line = box_coo_before
                box_coo = cv2.boxPoints(box_coo_before)
                box_coo = box_coo.astype('int')
            # print("In margin")

        else:
            box_coo_before = copy.deepcopy(box_coo)
            rect_for_line = box_coo
            box_coo = cv2.boxPoints(box_coo)
            box_coo = box_coo.astype('int')

        tuple_cov = [rect_for_line[0], list(rect_for_line[1]), rect_for_line[2]]

        # Change width and height if width is bigger
        if tuple_cov[1][0] > tuple_cov[1][1] :
            tuple_cov[1] = (tuple_cov[1][1], tuple_cov[1][0])
            tuple_cov[2] = tuple_cov[2] + 90

        if tuple_cov[1][0] < 50:
            return 0

        divide = int(tuple_cov[1][1] / length)

        # Length margin
        if length * 0.8 < tuple_cov[1][1] / divide < length * 1.2:
            length = tuple_cov[1][1] / divide

        while True:
            if tuple_cov[1][1] <= length * 2:
                if divide % 2 == 0:  # Even case
                    tuple_cov[1] = (tuple_cov[1][0], 0)
                    tuple_cov = tuple(tuple_cov)
                    # print(tuple_cov) Debug code for length
                    box_for_line = cv2.boxPoints(tuple_cov)
                    box_for_line = box_for_line.astype('int')

                    # img_box = cv2.drawContours(image, [box_for_line], -1, (0, 255, 0), 2)
                    img_box = cv2.drawContours(image, [box_for_line], 0, color, 2)
                    tuple_cov = [tuple_cov[0], list(tuple_cov[1]), tuple_cov[2]]
                    break
                else:
                    break

            tuple_cov[1] = (tuple_cov[1][0], tuple_cov[1][1] - length * 2)
            tuple_cov = tuple(tuple_cov)
            # print(tuple_cov) Debug code for length
            box_for_line = cv2.boxPoints(tuple_cov)
            box_for_line = box_for_line.astype('int')

            # img_box = cv2.drawContours(image, [box_for_line], -1, (0, 255, 0), 2)
            img_box = cv2.drawContours(image, [box_for_line], 0, color, 2)
            tuple_cov = [tuple_cov[0], list(tuple_cov[1]), tuple_cov[2]]

        image = cv2.drawContours(image, [box_coo], 0, color, 3)

    except:
        image = cv2.drawContours(image, [box_coo], 0, color, 3)

    return image

"""
def distance2pixel(FRAME_WIDTH):
    # calculate distance to use marker : how about import initialize sequence?
    # return d2pixel : get guide length and convert it
    # get image and apply gaussian filter or use 'noiseless' image just for initialize.
    # findcontours can find length between camera to marker.
"""
