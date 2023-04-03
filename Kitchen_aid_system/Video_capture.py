import cv2
import os

dir_result = os.path.dirname(os.path.abspath(__file__))

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# frame_rate = video.get(5)
captured_video = cv2.VideoWriter(dir_result + '\\captured_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (FRAME_WIDTH, FRAME_HEIGHT))

while True:
    ret, frame = video.read()
    captured_video.write(frame)
    cv2.imshow("Video_Frame", frame)
    if cv2.waitKey(1) > 0:
        break

video.release()
captured_video.release()
cv2.destroyAllWindows()