import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def FitLDA(data, labels, n_discriminant_variables = 2):
    # Step 1: Compute the Mean Vectors
    unique_classes = np.unique(labels)
    mean_vectors = []

    for cl in unique_classes:
        mean_vectors.append(np.mean(data[labels == cl], axis=0))

    # Step 2: Compute the Scatter Matrices
    # Within-Class Scatter Matrix
    S_W = np.zeros((data.shape[1], data.shape[1]))
    for cl, mv in zip(unique_classes, mean_vectors):
        class_sc_mat = np.zeros((data.shape[1], data.shape[1]))  # Scatter matrix for every class
        for row in data[labels == cl]:
            row, mv = row.reshape(data.shape[1], 1), mv.reshape(data.shape[1], 1)  # Make column vectors
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat
    
    # Between-Class Scatter Matrix
    overall_mean = np.mean(data, axis=0)
    S_B = np.zeros((data.shape[1], data.shape[1]))
    for cl, mean_vec in zip(unique_classes, mean_vectors):
        n = data[labels == cl].shape[0]
        mean_vec = mean_vec.reshape(data.shape[1], 1)
        overall_mean = overall_mean.reshape(data.shape[1], 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    # Step 3: Solve the Generalized Eigenvalue Problem for S_W^-1 S_B
    S_W_inv = np.linalg.inv(S_W)
    mat = S_W_inv.dot(S_B)

    eigenvalues, eigenvectors = np.linalg.eig(mat)

    # Sort the eigenvectors by decreasing eigenvalues
    eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
    # Sort the tuples by eigenvalue in descending order
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

    # Extract the eigenvectors for the n_discriminant_variables highest eigenvalues
    W = np.hstack([eigen_pairs[i][1].reshape(data.shape[1], 1) for i in range(n_discriminant_variables)])

    return W

def TransformLDA(data, eigenPairs, show_graph = False, labels = None):
    # Apply the LDA transformation
    transformed_data = data.dot(eigenPairs)

    if show_graph:
        # Get unique classes and their corresponding colors
        unique_classes = np.unique(labels)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))
        labelNames = { 0:"T-shirt/top", 1:"Trousers", 2:"Pullover", 3:"Dress", 4:"Shirt"}

        # Create a scatter plot with colors
        plt.figure(figsize=(10, 8))
        for i, unique_class in enumerate(unique_classes):

            class_mask = (labels == unique_class)
            plt.scatter(transformed_data[class_mask, 0], transformed_data[class_mask, 1], 
                        color=colors[i], label=f'Class {labelNames[unique_class]}', alpha=0.5)

        plt.xlabel('Linear Discriminant 1')
        plt.ylabel('Linear Discriminant 2')
        plt.title('LDA - Projection onto the first 2 linear discriminants')
        plt.legend()
        plt.show()

    return transformed_data

def VerifyLDA(data, labels, transformed_data, n_discriminant_variables=2):
    # Initialize sklearn's LDA
    lda = LinearDiscriminantAnalysis(n_components=n_discriminant_variables)

    # Fit sklearn's LDA model
    lda.fit(data, labels)
    sklearn_transformed_data = lda.transform(data)

    # Get unique classes and their corresponding colors
    unique_classes = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))
    labelNames = { 0:"T-shirt/top", 1:"Trousers", 2:"Pullover", 3:"Dress", 4:"Shirt"}

    # Create figure and axes for the subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Plot for Manual LDA
    for i, unique_class in enumerate(unique_classes):
        class_mask = (labels == unique_class)
        axes[0].scatter(transformed_data[class_mask, 0], transformed_data[class_mask, 1], 
                        color=colors[i], label=f'Class {labelNames[unique_class]}', alpha=0.5)

    axes[0].set_xlabel('Linear Discriminant 1')
    axes[0].set_ylabel('Linear Discriminant 2')
    axes[0].set_title('Manual LDA')
    axes[0].legend()

    # Plot for Sklearn LDA
    for i, unique_class in enumerate(unique_classes):
        class_mask = (labels == unique_class)
        axes[1].scatter(sklearn_transformed_data[class_mask, 0], sklearn_transformed_data[class_mask, 1], 
                        color=colors[i], label=f'Class {labelNames[unique_class]}', alpha=0.5)

    axes[1].set_xlabel('Linear Discriminant 1')
    axes[1].set_ylabel('Linear Discriminant 2')
    axes[1].set_title('Sklearn LDA')
    axes[1].legend()

    plt.show()
