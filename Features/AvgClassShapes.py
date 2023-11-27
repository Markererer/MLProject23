import numpy as np

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