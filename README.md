# Clothing Image Classification

This project uses machine learning to classify greyscale images of clothing into five categories: T-shirt/Top, Trousers, Pullover, Dress, and Shirt. Various feature extraction methods, dimensionality reduction techniques, and classifiers were implemented to optimize accuracy. The project also involved manually implementing Linear Discriminant Analysis (LDA) and Naive Bayes classifiers using Python and NumPy.

## Table of Contents
- [Abstract](#abstract)
- [Dataset Description](#dataset-description)
- [Methods and Techniques](#methods-and-techniques)
- [Model Evaluation](#model-evaluation)
- [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)
- [Appendix](#appendix)

## Abstract
This project classifies images of clothing from a 15,000-image dataset by Zalando. It compares three classifiers—Naive Bayes, Support Vector Machines (SVM), and Neural Networks—combined with feature extraction and dimensionality reduction techniques.

## Dataset Description
- **Total Images**: 15,000 greyscale (28x28 pixels)
- **Training Set**: 10,000 images
- **Testing Set**: 5,000 images
- **Classes**: T-shirt/Top, Trousers, Pullover, Dress, and Shirt

## Methods and Techniques
### Feature Extraction
- **Eigenshapes**: Used PCA to capture shape variations.
- **Pixel Differences**: Calculated differences from each class's mean image.
- **Histogram of Oriented Gradients (HOG)**: Captured structural patterns.
- **Black Pixel Ratio Along Y-Axis**: Targeted features like the trouser leg gap.

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Simplified and optimized classification.
- **Linear Discriminant Analysis (LDA)**: Custom implementation enhanced class separability.

### Classification Models
1. **Naive Bayes**: Modeled classification using Gaussian assumptions.
2. **Support Vector Machines (SVM)**: RBF kernel used with `sklearn` for high accuracy.
3. **Neural Network**: Built with TensorFlow to analyze deeper features.

## Model Evaluation
- **Accuracy**: Each classifier was evaluated on the test dataset.
- **Cross-validation**: Custom implementations were validated against `sklearn` and `TensorFlow` models.

## Results and Discussion
- **Best Performing Models**: SVM with RBF kernel and Neural Network.
- **Naive Bayes**: Fast but affected by feature dependencies.

## Conclusion
The SVM and Neural Network classifiers achieved high accuracy, with SVM performing best overall. The project highlights the strengths of SVM and neural networks in image classification, while Naive Bayes offered a faster alternative.

## Appendix
See the full project report in the `docs/` folder for additional analysis and figures.
