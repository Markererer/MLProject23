import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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