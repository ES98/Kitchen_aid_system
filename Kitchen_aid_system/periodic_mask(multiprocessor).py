import cv2
import math
import os
import RCNN_function as RCNN
import Init_model as Imodel
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import time
from multiprocessing import Process, Queue, Event

Imodel.RCNN_init()


def Img_Process(q1, q2, flg1, flg2, kill):

    print('Img Process Ready')
    while True:

        if flg1.is_set():
            if kill.is_set():
                break
            img = q1.get()
            print('Img process shape', np.shape(img))
            __r__ = Imodel.Make_Mask(img)  # make mask for frame
            print('scores', __r__['scores'])

            q2.put(__r__)
            # q2.put(1)
            flg2.set()

            # time.sleep(1)


def Img_Show(q1, q2, flg1, flg2, kill):

    dir_result = os.path.dirname(os.path.abspath(__file__))
    video = cv2.VideoCapture('Carrot.mp4')
    cv2.startWindowThread()

    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720 # for video capture

    frame_rate = video.get(5)
    print(frame_rate)
    i = 0
    j = 1
    mask = 0
    get_mask = 0

    cut_len = 65

    start_watch = 0
    stop_watch = 0
    while True:

        frameId = video.get(cv2.CAP_PROP_FRAME_COUNT)
        ret, frame = video.read()
        print(j)
        start_watch = time.time()
        captured_picture = dir_result + "\\captured_image\\" + str(i) + ".png"

        if flg2.is_set() :
            get_mask = q2.get()
            if get_mask['rois'].shape[0] is not 0:
                mask = get_mask

            flg1.clear()
            flg2.clear()
            print('data received')

        if not flg1.is_set():
            q1.put(frame)
            flg1.set()
            print('flg1 is set')

        if mask == 0:
            # print(mask['class_ids'])
            masked_frame = frame
            print('no masked frame')

        else:
            masked_frame = RCNN.make_masked_image(frame, mask['rois'], mask['masks'], mask['class_ids'],
                                                  cut_len, mask['scores']) # Apply mask at frame
            print('mask scores', mask['scores'])

        if type(masked_frame) == int:
            cv2.imwrite(captured_picture, frame)
            # captured_video.write(frame)
            cv2.imshow("Video_Frame", frame.astype(np.uint8))
        else:
            cv2.imwrite(captured_picture, masked_frame.astype(np.uint8))
            # captured_video.write(frame.astype(np.uint8))
            cv2.imshow("Video_Frame", masked_frame.astype(np.uint8))

        i += 1
        j += 1
        stop_watch = time.time()
        print(round(stop_watch - start_watch, 2), 'Seconds')
        # time.sleep(0.1)
        if j > frame_rate:
            j = 1
        if cv2.waitKey(1) > 0:
            break

    video.release()
    cv2.destroyAllWindows()
    kill.set()
    flg1.set()
    flg2.set()

if __name__ == '__main__':

    q1 = Queue()
    q2 = Queue()

    flg1 = Event()
    flg2 = Event()
    kill = Event()

    camera_process = Process(target=Img_Show, args=(q1, q2, flg1, flg2, kill))
    CNN_process = Process(target=Img_Process, args=(q1, q2, flg1, flg2, kill))
    CNN_process.start()
    camera_process.start()

    camera_process.join()
    CNN_process.join()

    q1.close()
    q1.join_thread()
    q2.close()
    q2.join_thread()

"""
while loop, 
Camera process -> CNN process -> handshake is 1?
-> adapt mask

write communication example 
"""
# 480, 640, 3 / rows, cols, dimensions
