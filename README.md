Clothing Image Classification
This project involves using machine learning techniques to classify greyscale images of clothing items into five categories (T-shirt/Top, Trousers, Pullover, Dress, and Shirt). It was implemented with multiple feature extraction methods, dimensionality reduction techniques, and classification models to achieve optimal accuracy. Key goals included manual implementation of Linear Discriminant Analysis (LDA) and Naive Bayes classifiers using Python and NumPy.

Table of Contents
Abstract
Dataset Description
Methods and Techniques
Model Evaluation
Results and Discussion
Conclusion
Appendix
Abstract
This project classifies images of clothing from a 15,000-image dataset by Zalando. We implemented and compared three classifiers—Naive Bayes, Support Vector Machines (SVM), and Neural Networks—combined with feature extraction and dimensionality reduction techniques to maximize accuracy.

Dataset Description
The dataset contains 15,000 greyscale images (28x28 pixels), with 5,000 images for testing and 10,000 for training, equally distributed among the five classes. Each class represents a type of clothing item.

Methods and Techniques
Feature Extraction
Eigenshapes: Principal component analysis (PCA) was applied to capture shape variations across images.
Pixel Differences from Mean Image: Differences between each image and its class mean image were used for additional class distinctions.
Histogram of Oriented Gradients (HOG): Aimed to capture structure and patterns within images.
Ratio of Black Pixels Alongside the Y-Axis: Focused on identifying trousers by the distinctive gap.
Dimensionality Reduction
Principal Component Analysis (PCA): Reduced dimensionality to simplify and optimize classification.
Linear Discriminant Analysis (LDA): A custom implementation to further enhance separability between classes.
Classification Models
Naive Bayes: Implemented from scratch to model image classification using Gaussian assumptions.
Support Vector Machines (SVM): Leveraged the sklearn SVM module with RBF kernel for classification.
Neural Network: Built using TensorFlow for deeper analysis of image features.
Model Evaluation
Each classifier was evaluated for accuracy and robustness on the test dataset, with results cross-validated against known implementations to verify our custom Naive Bayes and LDA.

Results and Discussion
The SVM model performed best on the dataset, with the RBF kernel providing a high classification accuracy. The neural network also demonstrated strong performance, particularly after color normalization. The Naive Bayes classifier was effective but was more affected by feature interdependencies than the other models.

Conclusion
This project provided a comprehensive exploration of feature extraction, dimensionality reduction, and classification techniques in image processing. The SVM and Neural Network classifiers showed promising results, especially for complex images, while the Naive Bayes model offered a faster, albeit less accurate, approach.

Appendix
Refer to the full project report in the docs/ folder for additional plots, figures, and detailed analysis.
