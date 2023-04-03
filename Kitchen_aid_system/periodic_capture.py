import cv2
import math
import os


dir_result = os.path.dirname(os.path.abspath(__file__))

FRAME_WIDTH = 852
FRAME_HEIGHT = 480

video = cv2.VideoCapture('home.mp4')
# video = cv2.VideoCapture(0)
# video.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# frame_rate = video.get(5)
captured_video = cv2.VideoWriter(dir_result + '\\captured_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (FRAME_WIDTH, FRAME_HEIGHT))
frame_rate = video.get(5)
print(frame_rate)
i = 0
j = 1
while True:
    frameId = video.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, frame = video.read()
    captured_video.write(frame)
    cv2.imshow("Video_Frame", frame)
    print(j)
    # j += 1
    # if j % math.floor(frame_rate) == 0:
    if i < 10:
        captured_picture = dir_result + "\\captured_image\\" + "000" + str(i) + ".jpg"
    elif i < 100:
        captured_picture = dir_result + "\\captured_image\\" + "00" + str(i) + ".jpg"
    elif i < 1000:
        captured_picture = dir_result + "\\captured_image\\" + "0" + str(i) + ".jpg"
    else :
        captured_picture = dir_result + "\\captured_image\\" + str(i) + ".jpg"
    cv2.imwrite(captured_picture, frame)
    i += 1
    #    i += 1
    #    j = 1
    if cv2.waitKey(1) > 0:
        break

video.release()
captured_video.release()
cv2.destroyAllWindows()