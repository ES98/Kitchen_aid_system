import cv2
import math
import os
import shutil
import RCNN_function as RCNN
import Init_model as Imodel
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import time

Imodel.RCNN_init()

dir_result = os.path.dirname(os.path.abspath(__file__))

fx = 815.464
fy = 818.790
cx = 343.575
cy = 236.116
k1 = -0.043
k2 = 0.127
p1 = 0.004
p2 = -0.001 # Camera distortion parameter

mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist = np.array([k1, k2, p1, p2, 0])

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

video = cv2.VideoCapture(0)
# video = cv2.VideoCapture('Carrot3.mp4')
cv2.startWindowThread()
# video.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

captured_video = cv2.VideoWriter(dir_result + '\\captured_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                 (FRAME_WIDTH, FRAME_HEIGHT))
frame_rate = video.get(5)
print(frame_rate)
i = 0
j = 1
mask = 0
start_watch = 0
stop_watch = 0
RUN_RCNN = False

cut_len = 0

while True:

    timer = time.gmtime(time.time()).tm_sec
    if timer % 2 is 0:  # If Seconds are even
        with open("./sync_text.txt", 'r') as sync_file:
            sync = sync_file.readlines()
            if "False" in sync[0]:
                RUN_RCNN = False
                if os.path.isdir("./captured_image") is True:
                    shutil.rmtree("./captured_image")
                print("False")
                continue
            elif "True" in sync[0]:
                RUN_RCNN = True
                if os.path.isdir("./captured_image") is False:
                    os.mkdir("./captured_image")
                not_value, cut_len = sync[1].split(' = ')
                cut_len = int(cut_len)

    if RUN_RCNN is True:
        print("RCNN works")
        frameId = video.get(cv2.CAP_PROP_FRAME_COUNT)
        ret, frame_raw = video.read()
        frame = cv2.undistort(frame_raw, mtx, dist, None, mtx)
        # print(j)
        start_watch = time.time()

        __r__ = Imodel.Make_Mask(frame)  # make mask for frame
        captured_picture = dir_result + "\\captured_image\\" + str(i) + ".png"

        masked_frame, only_box_frame = RCNN.make_masked_image(frame, __r__['rois'], __r__['masks'], __r__['class_ids'],
                                               cut_len, __r__['scores']) # Apply mask at frame
        # Only box frame is for projection to plane. it has white background and bounding box
        if type(masked_frame) == int or type(only_box_frame) == int:
            cv2.imwrite(captured_picture, only_box_frame)
            # captured_video.write(frame)
            # cv2.imshow("Video_Frame", frame.astype(np.uint8))
        else:
            cv2.imwrite(captured_picture, only_box_frame.astype(np.uint8))
            # captured_video.write(frame.astype(np.uint8))
            # cv2.imshow("Video_Frame", masked_frame.astype(np.uint8))

        i += 1
        # j += 1
        stop_watch = time.time()
        print(round(stop_watch - start_watch, 2), 'Seconds')
        """
        if j > frame_rate:
            j = 1
        if cv2.waitKey(1) > 0:
            break

    video.release()
    captured_video.release()
    cv2.destroyAllWindows()
    """
    # 480, 640, 3 / rows, cols, dimensions
