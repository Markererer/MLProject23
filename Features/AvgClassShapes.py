import numpy as np
import matplotlib.pyplot as plt

def GetAvgClassShapes(images, labels):
    unique_labels = np.unique(labels)
    average_images = {}

    for label in unique_labels:
        class_images = images[labels == label]
        mean_image = np.mean(class_images.reshape(-1, 28, 28), axis=0)

        average_images[label] = mean_image

    return average_images

def GetDiffFromAvgShapes(image, average_images):
    differences = {}
    for label, mean_image in average_images.items():
        differences[label] = np.sum((image.reshape(28, 28) - mean_image) ** 2)

    return np.array(list(differences.values()))

def DisplayImages(images, cols=5, figsize=(15, 5)):
    n = len(images)
    rows = n // cols + int(n % cols > 0)
    labels = ["T-shirt/Top", "Trousers", "Pullover", "Dress", "Shirt"]

    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap='gray', interpolation='none')
        plt.axis('off')
        plt.title(f"Class {i} ({labels[i]})")

    plt.tight_layout()
    plt.show()