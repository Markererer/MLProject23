import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
file_path = r'fashion_train.npy'  # Replace with your .npy file path

# Now, let's load the dummy .npy file as if it was the actual dataset
data = np.load(file_path)

# Separate the images from the labels
images, labels = data[:, :-1], data[:, -1]

# Get the unique classes
unique_labels = np.unique(labels)

# Initialize a dictionary to hold the average images for each class
average_images = {}

# Calculate the average image for each class
for label in unique_labels:
    # Extract images for the current class
    class_images = images[labels == label]

    # Reshape images to 28x28 and calculate the mean image
    mean_image = np.mean(class_images.reshape(-1, 28, 28), axis=0)

    # Store the mean image in the dictionary
    average_images[label] = mean_image

# Display the average images for each class
#fig, axes = plt.subplots(1, len(unique_labels), figsize=(15, 2))
#for i, label in enumerate(unique_labels):
#    ax = axes[i]
#    ax.imshow(average_images[label], cmap='gray')
#    ax.axis('off')
#    ax.set_title(f'Class {label}')
#plt.show()

def classify_image(test_image, average_images):
    """
    Classify the image by comparing it with the average images.
    Args:
    - test_image: a 28x28 numpy array representing the image to be classified.
    - average_images: a dictionary with class labels as keys and average images as values.

    Returns:
    - predicted_label: the label of the class that the test image is predicted to belong to.
    """
    min_diff = float('inf')
    predicted_label = None

    # Iterate over each class's mean image and calculate the sum of squared differences
    for label, mean_image in average_images.items():
        diff = np.sum((test_image - mean_image) ** 2)
        if diff < min_diff:
            min_diff = diff
            predicted_label = label

    return predicted_label

# Load or select a test image (reshaping if necessary)
test_image_index = 1  # Change this index to test with different images
test_image = images[test_image_index].reshape(28, 28)

# Classify the test image
predicted_label = classify_image(test_image, average_images)

labelNames = { 0:"T-shirt/top", 1:"Trousers", 2:"Pullover", 3:"Dress", 4:"Shirt" }

# Show the test image and its predicted label
#plt.imshow(test_image, cmap='gray')
#plt.axis('off')
#plt.title(f'Predicted Label: {labelNames[predicted_label]}; Actual Label: {labelNames[labels[test_image_index]]}')
#plt.show()

# Load the test dataset
file_path = r'fashion_test.npy'
data = np.load(file_path)
images, labels = data[:, :-1], data[:, -1]

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Initialize lists to store the true and predicted labels
true_labels = []
predicted_labels = []

# Classify each image in the test set
for i in range(len(images)):
    predicted_label = classify_image(images[i].reshape(28, 28), average_images)
    predicted_labels.append(predicted_label)
    true_labels.append(labels[i])

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy}; Precision: {precision}; Recall: {recall}; F1: {f1}")