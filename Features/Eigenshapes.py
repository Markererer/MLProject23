import numpy as np
import matplotlib.pyplot as plt

def GetEigenShapes(images, n_components=50, plot=False):
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

    if plot:
        PlotEigenshapes(sorted_eigenvectors)

    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    
    return selected_eigenvectors

def PlotEigenshapes(eigenshapes, grid_size=(4, 4), fig_size=(10, 10)):
    fig, axes = plt.subplots(*grid_size, figsize=fig_size)
    eigenshapes_count = grid_size[0] * grid_size[1]

    # Normalize the eigenvectors globally
    global_min = eigenshapes.min()
    global_max = eigenshapes.max()

    for i in range(eigenshapes_count):
        ax = axes[i // grid_size[1], i % grid_size[0]]
        eigenshape = eigenshapes[:, i].reshape(28, 28)

        # Normalize the eigenvector for better visualization
        norm_eigenshape = (eigenshape - global_min) / (global_max - global_min)

        # Use 'gray' colormap and 'nearest' interpolation
        ax.imshow(norm_eigenshape, cmap='RdBu', interpolation='nearest')
        ax.axis('off')

    plt.show()