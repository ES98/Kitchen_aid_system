import cv2
import numpy as np
print("ok")
from skimage.data import horse
from matplotlib import pyplot as plt

img_test = cv2.imread('light.png')
img_raw = cv2.imread('light.png')
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
ret, img_raw = cv2.threshold(img_raw, 127, 255, cv2.THRESH_BINARY)
img = img_raw.copy().astype('uint8')
"""
# horse data
horse = horse().astype('uint8')
horse = np.ones(horse.shape) - horse
img = horse.copy().astype('uint8')
"""

# contour
contours, hierachy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#print(np.shape(contours)) #(1, 328, 1, 2)
#print(np.shape(contours[0]))#(328, 1, 2)

rect = cv2.minAreaRect(contours[0]) # return center coordinate and width, height, angle
rect_for_line = rect
print(rect_for_line)
box = cv2.boxPoints(rect)
box = box.astype('int') # coordinates for bounding box

# print(rect_for_line[1][0])
print(box)

length = 65
tuple_cov = 0

tuple_cov = [rect_for_line[0], list(rect_for_line[1]), rect_for_line[2]]

divide = int(tuple_cov[1][1] / length)
"""
if (divide % 2) == 0 :
    divide += 1
"""
if length * 0.8 < tuple_cov[1][1] / divide < length * 1.2 :
    length = tuple_cov[1][1] / divide

while True:
    if tuple_cov[1][1] <= length*2:
        if divide %2 == 0 : # 짝수인 경우
            tuple_cov[1] = (tuple_cov[1][0], 0)
            tuple_cov = tuple(tuple_cov)
            print(tuple_cov)
            box_for_line = cv2.boxPoints(tuple_cov)
            box_for_line = box_for_line.astype('int')

            img_box = cv2.drawContours(img_test, [box_for_line], -1, (0, 255, 0), 2)
            tuple_cov = [tuple_cov[0], list(tuple_cov[1]), tuple_cov[2]]
            break
        else :
            break

    # tuple_cov = [rect_for_line[0], list(rect_for_line[1]), rect_for_line[2]]
    tuple_cov[1] = (tuple_cov[1][0], tuple_cov[1][1] - length*2)
    tuple_cov = tuple(tuple_cov)
    print(tuple_cov)
    box_for_line = cv2.boxPoints(tuple_cov)
    box_for_line = box_for_line.astype('int')

    img_box = cv2.drawContours(img_test, [box_for_line], -1, (0, 255, 0),2)
    tuple_cov = [tuple_cov[0], list(tuple_cov[1]), tuple_cov[2]]

img_box = cv2.drawContours(img_test, [box], -1 ,(0, 255, 0),2)

plt.imshow(img_box, cmap = 'gray')
plt.axis('off')
plt.show()
