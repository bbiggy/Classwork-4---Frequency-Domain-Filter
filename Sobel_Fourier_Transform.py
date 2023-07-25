import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('kurisu.jpg', cv.IMREAD_GRAYSCALE)

# Create Sobel Filter X (Horizontal)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# Create Sobel Filter Y (Vertical)
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Fourier Transform of the input image
image_freq = np.fft.fft2(image)
image_freq_shifted = np.fft.fftshift(image_freq)

# find magnitude image 
image_Real = np.real(image_freq_shifted)
image_Imagine = np.imag(image_freq_shifted)
image_Magitude  = np.sqrt(image_Real**2 + image_Imagine**2)
image_Magitude = np.log(1+image_Magitude)
image_Magitude = cv.normalize(image_Magitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

# Fourier Transform of Sobel Filter X (Horizontal)
sobel_x_freq = np.fft.fft2(sobel_x, s=image.shape)
sobel_x_freq_shifted = np.fft.fftshift(sobel_x_freq)

# Fourier Transform of Sobel Filter Y (Vertical)
sobel_y_freq = np.fft.fft2(sobel_y, s=image.shape)
sobel_y_freq_shifted = np.fft.fftshift(sobel_y_freq)

# find magnitude sobel filter x
filter_Real_x = np.real(sobel_x_freq_shifted)
filter_Imagine_x = np.imag(sobel_x_freq_shifted)
filter_Magitude_x  = np.sqrt(filter_Real_x**2 + filter_Imagine_x**2)
filter_Magitude_x = np.log(1+filter_Magitude_x)

# find magnitude sobel filter y
filter_Real_y = np.real(sobel_y_freq_shifted)
filter_Imagine_y = np.imag(sobel_y_freq_shifted)
filter_Magitude_y  = np.sqrt(filter_Real_y**2 + filter_Imagine_y**2)
filter_Magitude_y = np.log(1+filter_Magitude_y)

# Element-wise multiplication (Dot Product) with Sobel X in the frequency domain
filtered_image_x_freq_shifted = image_freq_shifted * sobel_x_freq_shifted

# Element-wise multiplication (Dot Product) with Sobel Y in the frequency domain
filtered_image_y_freq_shifted = image_freq_shifted * sobel_y_freq_shifted

# Inverse Fourier Transform to get the filtered images back in Spatial Domain
filtered_image_x_freq_shifted = np.fft.ifftshift(filtered_image_x_freq_shifted)
filtered_image_y_freq_shifted = np.fft.ifftshift(filtered_image_y_freq_shifted)

filtered_image_x_spatial_domain = np.fft.ifft2(filtered_image_x_freq_shifted)
filtered_image_y_spatial_domain = np.fft.ifft2(filtered_image_y_freq_shifted)

# Take the magnitude of the complex-valued filtered images (optional)
filtered_image_x_magnitude = np.abs(filtered_image_x_spatial_domain)
filtered_image_y_magnitude = np.abs(filtered_image_y_spatial_domain)

# Normalize range [0, 255]
filter_Magitude_x = cv.normalize(filter_Magitude_x, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
filter_Magitude_y = cv.normalize(filter_Magitude_y, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
sobel_x = cv.normalize(filtered_image_x_magnitude, None, 0, 255, cv.NORM_MINMAX,cv.CV_8U)
sobel_y = cv.normalize(filtered_image_y_magnitude, None, 0, 255, cv.NORM_MINMAX,cv.CV_8U)

# Display the filtered images in Spatial Domain
cv.imshow("sobel_x_IFT",sobel_x)
cv.imshow("sobel_y_IFT",sobel_y)
cv.imshow("image_Magitude",image_Magitude)
cv.imshow('Sobel X (Horizontal)', filter_Magitude_x)
cv.imshow('Sobel Y (Vertical)', filter_Magitude_y)
cv.imwrite('sobel_x_IFT.png', sobel_x)
cv.imwrite('sobel_y_IFT.png', sobel_y)
cv.waitKey(0)
cv.destroyAllWindows()
