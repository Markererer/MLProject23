import numpy as np
import cv2
from skimage.feature import hog
from numpy.fft import fft

def ExtractContourFeatures(image):
    image = image.reshape((28, 28))
    image = np.uint8(image)
    _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    contour_areas = [cv2.contourArea(c) for c in contours]
    total_contour_area = sum(contour_areas)
    return np.array([num_contours, total_contour_area])

def ExtractHOGFeatures(image):
    image = image.reshape((28, 28))
    features, _ = hog(image, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True, feature_vector=True)
    return features

def ExtractFourierDescriptors(image):
    image = image.reshape((28, 28))
    image = np.uint8(image)
    _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        # Take the longest contour
        contour = max(contours, key=cv2.contourArea)
        contour = contour.flatten()
        # Compute the Fourier Transform of the contour
        fourier_result = fft(contour)
        # Return the absolute values of the first few coefficients
        return np.abs(fourier_result[:5])  # You can adjust the number of coefficients
    else:
        return np.zeros(5)  # Return zeros if no contours are found