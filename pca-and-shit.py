import numpy as np
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from numpy.fft import fft


# Function to extract contour features
def extract_contour_features(image):
    image = image.reshape((28, 28))
    image = np.uint8(image)
    _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    contour_areas = [cv2.contourArea(c) for c in contours]
    total_contour_area = sum(contour_areas)
    return [num_contours, total_contour_area]

# Function to extract HOG features
def extract_hog_features(image):
    image = image.reshape((28, 28))
    features, _ = hog(image, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True, feature_vector=True)
    return features

# Function to extract eigenvalues
def extract_eigenvalues(image):
    image = image.reshape((28, 28))
    covariance_matrix = np.cov(image, rowvar=False)
    eigenvalues, _ = np.linalg.eig(covariance_matrix)
    return eigenvalues.real

def extract_fourier_descriptors(image):
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


# Load the data
data = np.load('MLProject23/fashion_train.npy')
X = data[:, :-1]  # Features
y = data[:, -1]   # Labels

# Extract features
contour_features = np.array([extract_contour_features(image) for image in X])
hog_features = np.array([extract_hog_features(image) for image in X])
eigenvalue_features = np.array([extract_eigenvalues(image) for image in X])
fourier_features = np.array([extract_fourier_descriptors(image) for image in X])

# Combine the features
combined_features = np.hstack((contour_features, hog_features, eigenvalue_features, fourier_features))

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(combined_features)

# Apply LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(combined_features, y)

# Visualization
plt.figure(figsize=(12, 6))

# PCA Plot
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title('PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# LDA Plot
plt.subplot(1, 2, 2)
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.title('LDA Projection')
plt.xlabel('LD 1')
plt.ylabel('LD 2')

plt.show()
