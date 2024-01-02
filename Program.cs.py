import numpy as np
from Features import Eigenshapes
from Features import AvgClassShapes
from Features import ExtractFeatures
from Dimensionality import LDA
from Dimensionality import PCA
from Classifiers import NaiveBayes
from Classifiers import SVM
from sklearn.preprocessing import StandardScaler
#from feature_distribution import assess_feature_distribution
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

def TransformAndExtractFeatures(images, eigenshapes, average_shapes):
    transformed_features = []

    for image in images:
        flattened_image = image.flatten()
        transformed_image = np.dot(flattened_image, eigenshapes)

        ## Extract other features
        features = []

        # Pixel differences between this image and the average shape of each class 
        features.append(AvgClassShapes.GetDiffFromAvgShapes(image, average_shapes))
        # Contour features
        #features.append(ExtractFeatures.ExtractContourFeatures(image))
        # Histogram of Oriented Gradients (HOG) features
        features.append(ExtractFeatures.ExtractHOGFeatures(image))
        # Fourier descriptors
        #features.append(ExtractFeatures.ExtractFourierDescriptors(image))
        # Ratio of black pixels in the middle (specifically for classifying trousers)
        features.append(ExtractFeatures.CountBlackPixelsInMiddle(image))

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

    # Extract features and fit the data to match the Eigenshapes extracted (784 pixels transformed into 50 principal components)
    features_train = TransformAndExtractFeatures(X_train, eigenshapes, average_shapes)
    features_test = TransformAndExtractFeatures(X_test, eigenshapes, average_shapes)

    # Scale features so that they all have the same scale (idk how to explain it, think how one value goes from 0-1 and another can go to 7000, this makes it so they all go from x-y)
    scaler = StandardScaler()
    scaler.fit(features_train)

    X_train = scaler.transform(features_train)
    X_test = scaler.transform(features_test)

    # PCA - All features scaled down to the 10 highest variance principal components
    pca = PCA.FitPCA(X_train)
    X_train = PCA.TransformPCA(X_train, pca)
    X_test = PCA.TransformPCA(X_test, pca)

    # Assess feature distribution and capture standard deviations
    #feature_std_devs = assess_feature_distribution(X_train)

    # LDA - The 10 principal components are scaled down to 2 linear discriminant variables (as per project requirements)
    #ldaEigenPairs = LDA.FitLDA(X_train, y_train)
    #X_train_LDA = LDA.TransformLDA(X_train, ldaEigenPairs)
    #X_test_LDA = LDA.TransformLDA(X_test, ldaEigenPairs)
    #LDA.VerifyLDA(X_train, y_train, X_train_LDA, 2) #Visually the same, except flipped around because of eigenvector direction
    
    ## Comment out the ones you're not using to save time
    # Manual Naive Bayes classifier that uses the first two linear discriminant variables as features
    #priors, stats = NaiveBayes.FitClassifier(X_train_LDA, y_train)
    #y_pred = [NaiveBayes.Classify(instance, priors, stats) for instance in X_test_LDA]

    #NaiveBayes.EvaluateClassifier(y_test, y_pred)

    # Sklearn implementation to compare the results to
    #gnb = GaussianNB()
    #gnb.fit(np.real(X_train_LDA), y_train)
    #y_pred = gnb.predict(np.real(X_test_LDA))
    #f1 = f1_score(y_test, y_pred, average='weighted')
    #print(f"F1 Score: {f1}")

    # SVM classifier
    svmClassifier = SVM.FitClassifier(np.real(X_train), y_train, C=2)
    y_pred = svmClassifier.predict(np.real(X_test))

    SVM.EvaluateClassifier(y_test, y_pred)

if __name__ == "__main__":
    Main()