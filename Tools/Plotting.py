from Classifiers.NeuralNetwork import neural_network
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


model, history = neural_network()
def plot_history(history):
    # Plot the training and validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation (Test) Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return

def plot_confusion_matrix(model, X_test, y_test):
        
    # Use the model to make predictions on the test dataset
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    y_true_classes = np.argmax(y_test, axis=1)  # True class labels

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()  
    return

