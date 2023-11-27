import numpy as np
from Features import Eigenshapes
from Features import AvgClassShapes
from Features import ExtractFeatures
from sklearn.preprocessing import StandardScaler

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

        features = np.concatenate(features)
        combined_features = np.concatenate([transformed_image, features])
        transformed_features.append(combined_features)
        
    return np.array(transformed_features)

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

    features_train = scaler.transform(features_train)
    features_test = scaler.transform(features_test)

    print(np.shape(features_train))
    print(np.shape(features_test))

    # TODO: Perform PCA and LDA and show graphs of the first two components
    # TODO: Manual Naive Bayes classifier that uses the first two linear discriminant variables as features

if __name__ == "__main__":
    Main()