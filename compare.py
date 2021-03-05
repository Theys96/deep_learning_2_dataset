import cv2
import numpy as np

img1 = cv2.imread("dataset\\00000.jpg")
img2 = cv2.imread("dataset\\00001.jpg")

print(np.abs(img1 - img2))
cv2.imwrite("compare.png", np.abs(img1 - img2))

