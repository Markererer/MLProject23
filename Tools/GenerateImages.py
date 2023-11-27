import numpy as np
import matplotlib.pyplot as plt
import os

# Load the .npy file
file_path = 'MLProject23/fashion_train.npy'  # Replace with your .npy file path

# Load the data
data = np.load(file_path)

# Separate the images from the labels
images, labels = data[:, :-1], data[:, -1]

# Reshape the images for display
images_reshaped = images.reshape(-1, 28, 28)

# Create a directory to store the images, change this to your preferred location
export_dir = 'exported_images'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# Create subdirectories for each class and save images
for class_id in np.unique(labels):
    class_dir = os.path.join(export_dir, f'class_{int(class_id)}')
    os.makedirs(class_dir, exist_ok=True)

    # Filter images for this class
    class_images = images_reshaped[labels == class_id]

    # Save the first N images of this class
    for i in range(min(1000, len(class_images))):
        image_path = os.path.join(class_dir, f'image_{i+1}.jpg')
        plt.imsave(image_path, class_images[i], cmap='gray')
