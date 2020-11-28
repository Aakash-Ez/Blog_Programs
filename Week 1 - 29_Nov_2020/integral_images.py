import cv2
import numpy as np

image = cv2.imread("regular_image.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rows,cols=image.shape

dest = np.zeros((rows,cols),np.float32)

integral_image = cv2.integral(image,dest,cv2.CV_32F)
cv2.normalize(integral_image, integral_image, 0, 1, cv2.NORM_MINMAX) 

cv2.imshow('Original Image',image)
cv2.imshow("Integral Image",integral_image)
cv2.waitKey()