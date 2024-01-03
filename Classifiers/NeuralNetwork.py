import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def neural_network():

    # Load the data
    training_data = np.load('fashion_train.npy')
    test_data = np.load('fashion_test.npy')

    # Normalize the pixel values
    X_train = training_data[:, :-1] / 255.0
    X_test = test_data[:, :-1] / 255.0

    # Reshape data to fit the model
    X_train = X_train.reshape((-1, 28, 28, 1))
    X_test = X_test.reshape((-1, 28, 28, 1))

    # Extract labels and convert to one-hot encoding
    y_train = to_categorical(training_data[:, -1], num_classes=5)
    y_test = to_categorical(test_data[:, -1], num_classes=5)

    # Define the model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model and record the history
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return model, history


neural_network()
