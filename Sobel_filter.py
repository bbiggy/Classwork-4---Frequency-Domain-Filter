import cv2 as cv
import numpy as np


img = cv.imread('kurisu.jpg', cv.IMREAD_GRAYSCALE)

#laplacian = cv.Laplacian(img, cv.CV_64F)
sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

# laplacian = cv.normalize(laplacian, None, 0, 255, cv.NORM_MINMAX,cv.CV_8U)
sobel_x = cv.normalize(sobel_x, None, 0, 255, cv.NORM_MINMAX,cv.CV_8U)
sobel_y = cv.normalize(sobel_y, None, 0, 255, cv.NORM_MINMAX,cv.CV_8U)

#cv.imwrite('laplacian.png', laplacian)
cv.imshow("sobel_x_Spatial ",sobel_x)
cv.imshow("sobel_y_Spatial ",sobel_y)
cv.imwrite('sobel_x_Spatial.png', sobel_x)
cv.imwrite('sobel_y_Spatial.png', sobel_y)
cv.waitKey(0)
cv.destroyAllWindows()