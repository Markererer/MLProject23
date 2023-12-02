import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def _CalculatePriors(y_train):
    classes, class_counts = np.unique(y_train, return_counts=True)
    total_count = y_train.shape[0]
    priors = class_counts / total_count
    return dict(zip(classes, priors))

def _CalculateClassStatistics(X_train, y_train):
    classes = np.unique(y_train)
    stats = {}
    for cls in classes:
        features_in_class = X_train[y_train == cls]
        stats[cls] = {
            "mean": features_in_class.mean(axis=0),
            "var": features_in_class.var(axis=0)
        }
    return stats

def _PDFFunction(x, mean, var):
    const = 1 / np.sqrt(2 * np.pi * var)
    prob = const * np.exp(-((x - mean) ** 2) / (2 * var))
    return prob

def Classify(instance, priors, stats):
    posteriors = []
    for cls, class_stats in stats.items():
        prior = np.log(priors[cls])
        class_likelihood = np.sum(np.log(_PDFFunction(instance, class_stats['mean'], class_stats['var'])))
        posterior = prior + class_likelihood
        posteriors.append(posterior)
    return np.argmax(posteriors)

def FitClassifier(X_train, y_train):
    priors = _CalculatePriors(y_train)
    stats = _CalculateClassStatistics(X_train, y_train)
    return priors, stats

def EvaluateClassifier(y_true, y_pred):
    # Calculate the metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Print the metrics
    print("Naive Bayes classifier:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")