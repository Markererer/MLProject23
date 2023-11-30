import numpy as np
from Features import Eigenshapes
from Features import AvgClassShapes
from Features import ExtractFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def TransformAndExtractFeatures(images, eigenshapes, average_shapes):
    transformed_features = []

    for image in images:
        flattened_image = image.flatten()
        transformed_image = np.dot(flattened_image, eigenshapes)

        # Extract other features
        features = []

        # Pixel differences between this image and the average shape of each class 
        features.append(AvgClassShapes.GetDiffFromAvgShapes(image, average_shapes))
        # Contour features
        features.append(ExtractFeatures.ExtractContourFeatures(image))
        # Histogram of Oriented Gradients (HOG) features
        features.append(ExtractFeatures.ExtractHOGFeatures(image))
        # Fourier descriptors
        features.append(ExtractFeatures.ExtractFourierDescriptors(image))
        # Ratio of black pixels in the middle (specifically for classifying trousers)
        features.append(ExtractFeatures.CountBlackPixelsInMiddle(image))

        features = np.concatenate(features)
        combined_features = np.concatenate([transformed_image, features])
        transformed_features.append(combined_features)
        
    return np.array(transformed_features)

def FitPCA(data, n_principal_components = 10):
    pca = PCA(n_components=n_principal_components)
    X_train_pca = pca.fit(data)

    return X_train_pca

def TransformPCA(data, pca, show_graph = False, labels = None):
    data = pca.transform(data)

    # Get unique classes and their corresponding colors
    unique_classes = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))
    labelNames = { 0:"T-shirt/top", 1:"Trousers", 2:"Pullover", 3:"Dress", 4:"Shirt"}

    if show_graph:
        # Create a scatter plot with colors
        plt.figure(figsize=(10, 8))
        for i, unique_class in enumerate(unique_classes):
            class_mask = (labels == unique_class)
            plt.scatter(data[class_mask, 0], data[class_mask, 1], 
                        color=colors[i], label=f'Class {labelNames[unique_class]}', alpha=0.5)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA - First Two Principal Components by Class')
        plt.legend()
        plt.show()

    return data

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

    # Compare the results
    #comparison = cosine_similarity(transformed_data, sklearn_transformed_data)

    # Calculate average similarity (optional)
    #average_similarity = np.mean(comparison)

    # Get unique classes and their corresponding colors
    unique_classes = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))
    labelNames = { 0:"T-shirt/top", 1:"Trousers", 2:"Pullover", 3:"Dress", 4:"Shirt"}

    # Create a scatter plot with colors
    plt.figure(figsize=(10, 8))
    for i, unique_class in enumerate(unique_classes):

        class_mask = (labels == unique_class)
        plt.scatter(sklearn_transformed_data[class_mask, 0], sklearn_transformed_data[class_mask, 1], 
                    color=colors[i], label=f'Class {labelNames[unique_class]}', alpha=0.5)

    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.title('LDA - Projection onto the first 2 linear discriminants')
    plt.legend()
    plt.show()

    #print(comparison)
    #print(average_similarity)

def Main():
    train_path = r'fashion_train.npy'
    test_path = r'fashion_test.npy'
    data = np.load(train_path)
    X_train, y_train = data[:, :-1], data[:, -1]
    data = np.load(test_path)
    X_test, y_test = data[:, :-1], data[:, -1]

    # TODO: A point could be made to save these into a file instead of calculating them every time
    eigenshapes = Eigenshapes.GetEigenShapes(X_train)
    average_shapes = AvgClassShapes.GetAvgClassShapes(X_train, y_train)

    features_train = TransformAndExtractFeatures(X_train, eigenshapes, average_shapes)
    features_test = TransformAndExtractFeatures(X_test, eigenshapes, average_shapes)

    scaler = StandardScaler()
    scaler.fit(features_train)

    X_train = scaler.transform(features_train)
    X_test = scaler.transform(features_test)

    pca = FitPCA(X_train)
    X_train = TransformPCA(X_train, pca)

    ldaEigenPairs = FitLDA(X_train, y_train)
    X_train_LDA = TransformLDA(X_train, ldaEigenPairs, show_graph=True, labels=y_train)

    VerifyLDA(X_train, y_train, X_train_LDA, 2)
    # TODO: Manual Naive Bayes classifier that uses the first two linear discriminant variables as features

if __name__ == "__main__":
    Main()