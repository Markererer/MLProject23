import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Load the .npy file
file_path = r'C:\Users\knedl\Desktop\Uni Things\3 Machine Learning\Project datasets\fashion_train.npy'  # Replace with your .npy file path

# Now, let's load the dummy .npy file as if it was the actual dataset
data = np.load(file_path)

# Separate the images from the labels
images, labels = data[:, :-1], data[:, -1]

# Reshape the images for display
images_reshaped = images.reshape(-1, 28, 28)

# Display the shape of the dataset and the first few images
data_shape = images_reshaped.shape

# Show the first few images in the dataset
#fig, axes = plt.subplots(1, 5, figsize=(10, 2))
#for i, ax in enumerate(axes):
#    ax.imshow(images_reshaped[i], cmap='gray')
#    ax.axis('off')
#plt.show()

#data_shape, labels[:5]  # Display the first five labels for reference

export_dir = 'exported_images'
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)

# Recreate the directory
os.makedirs(export_dir)

# Save the first 1000 images as .jpg files
for i in range(1000):
    image_path = os.path.join(export_dir, f'image_{i+1}.jpg')
    plt.imsave(image_path, images_reshaped[i], cmap='gray')