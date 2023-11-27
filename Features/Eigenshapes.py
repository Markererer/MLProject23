import numpy as np

def GetEigenShapes(images, n_components=50):
    flattened_images = [img.flatten() for img in images]
    data_matrix = np.vstack(flattened_images)

    # Center the data
    mean_image = np.mean(data_matrix, axis=0)
    centered_data_matrix = data_matrix - mean_image

    covariance_matrix = np.cov(centered_data_matrix, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvectors based on eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]

    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    
    return selected_eigenvectors