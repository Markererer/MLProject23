import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
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

# Function to extract Local Binary Pattern (LBP) features
def extract_lbp_features(image):
    image = image.reshape((28, 28))
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Function to extract Pixel Intensity Statistics
def extract_intensity_stats(image):
    mean = np.mean(image)
    std = np.std(image)
    skewness = skew(image.reshape(-1))
    kurt = kurtosis(image.reshape(-1))
    return [mean, std, skewness, kurt]

def count_pixel_intensities(image, lower_bound, upper_bound):
    image = image.reshape((28, 28))
    count = np.sum((image >= lower_bound) & (image <= upper_bound))
    return count

# Example usage: Counting near-white and near-black pixels
# You can adjust the thresholds as per your requirement
def extract_color_features(image):
    near_white_count = count_pixel_intensities(image, 200, 255)  # Light pixels
    near_black_count = count_pixel_intensities(image, 0, 50)      # Dark pixels
    return [near_white_count, near_black_count]




# Load the data
data = np.load('MLProject23/fashion_train.npy')
X = data[:, :-1]  # Features
y = data[:, -1]   # Labels

# Extract features
contour_features = np.array([extract_contour_features(image) for image in X])
hog_features = np.array([extract_hog_features(image) for image in X])
eigenvalue_features = np.array([extract_eigenvalues(image) for image in X])
fourier_features = np.array([extract_fourier_descriptors(image) for image in X])
lbp_features = np.array([extract_lbp_features(image) for image in X])
intensity_stats_features = np.array([extract_intensity_stats(image) for image in X])
# Extract color features
color_features = np.array([extract_color_features(image) for image in X])

# Combine the features
combined_features = np.hstack((contour_features, hog_features, eigenvalue_features, fourier_features, color_features))

# Scale the features
scaler = StandardScaler()
scaled_combined_features = scaler.fit_transform(combined_features)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(combined_features)

# Apply LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(combined_features, y)

# Calculate mean points for each class in PCA and LDA space
mean_points_pca = np.array([np.mean(X_pca[y == label, :], axis=0) for label in np.unique(y)])
mean_points_lda = np.array([np.mean(X_lda[y == label, :], axis=0) for label in np.unique(y)])

# Visualization
plt.figure(figsize=(18, 6))

# PCA Plot
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.5)
plt.title('PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# LDA Plot
plt.subplot(1, 3, 2)
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, alpha=0.5)
plt.title('LDA Projection')
plt.xlabel('LD 1')
plt.ylabel('LD 2')

# Mean Points Plot for PCA
plt.subplot(1, 3, 3)
plt.scatter(mean_points_pca[:, 0], mean_points_pca[:, 1], c=np.unique(y))
for i, label in enumerate(np.unique(y)):
    plt.text(mean_points_pca[i, 0], mean_points_pca[i, 1], str(label), ha='center', va='center')
plt.title('Mean Points in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()