import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from Dimensionality.PCA import FitPCA, TransformPCA
from Dimensionality.LDA import FitLDA, TransformLDA
import pandas as pd

# Load the data
training_data = np.load('fashion_train.npy')
test_data = np.load('fashion_test.npy')

training_data_real = np.real(training_data)
test_data_real = np.real(test_data)

# Normalize the pixel values
X_train = training_data_real[:, :-1] / 255.0
X_test = test_data_real[:, :-1] / 255.0

# Apply PCA
pca = FitPCA(X_train)
X_train_pca = TransformPCA(X_train, pca)
X_test_pca = TransformPCA(X_test, pca)

# Apply LDA
lda = FitLDA(X_train, training_data[:, -1])
X_train_lda = TransformLDA(X_train, lda)
X_test_lda = TransformLDA(X_test, lda)

# Define the model creation function
def create_model(input_shape):
    model = Sequential([
        Flatten(input_shape=(input_shape,)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train and evaluate the model
performance_data = []
def evaluate_model(model, X_train, y_train, X_test, y_test, description):
    model.fit(X_train, y_train, epochs=10)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Evaluation on {description} Data: Loss = {loss}, Accuracy = {accuracy}")
    performance_data.append({
        'Description': description,
        'Loss': loss,
        'Accuracy': accuracy
    })

# Load your data and preprocess (Assuming this part is already done)
# X_train, X_test, X_train_pca, X_test_pca, X_train_lda, X_test_lda, training_data, test_data

# Concatenate features for different model inputs
X_train_og_pca = np.concatenate((X_train, X_train_pca), axis=1)
X_test_og_pca = np.concatenate((X_test, X_test_pca), axis=1)
X_train_og_lda = np.concatenate((X_train, X_train_lda), axis=1)
X_test_og_lda = np.concatenate((X_test, X_test_lda), axis=1)
X_train_combined = np.concatenate((X_train, X_train_pca, X_train_lda), axis=1)
X_test_combined = np.concatenate((X_test, X_test_pca, X_test_lda), axis=1)

# Evaluate models on different data combinations
evaluate_model(create_model(X_train.shape[1]), X_train, training_data[:, -1], X_test, test_data[:, -1], "Original")
evaluate_model(create_model(X_train_og_pca.shape[1]), X_train_og_pca, training_data[:, -1], X_test_og_pca, test_data[:, -1], "OG + PCA")
evaluate_model(create_model(X_train_og_lda.shape[1]), X_train_og_lda, training_data[:, -1], X_test_og_lda, test_data[:, -1], "OG + LDA")
evaluate_model(create_model(X_train_combined.shape[1]), X_train_combined, training_data[:, -1], X_test_combined, test_data[:, -1], "OG + PCA + LDA")

# Save the performance data to a CSV file
performance_df = pd.DataFrame(performance_data)
performance_file_path = 'model_performance.csv'
performance_df.to_csv(performance_file_path, index=False)

print(f"Performance data saved to {performance_file_path}")