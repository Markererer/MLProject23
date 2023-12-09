import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def assess_feature_distribution(features):
    num_features = features.shape[1]
    feature_std_devs = []

    for i in range(num_features):
        feature = features[:, i]

        # Perform the Shapiro-Wilk test for normality
        _, p_value = stats.shapiro(feature)

        print(f'Feature {i + 1} Shapiro-Wilk p-value: {p_value}')

        if p_value < 0.05:
            print(f'Feature {i + 1} is not normally distributed.')
        else:
            print(f'Feature {i + 1} appears to be normally distributed.')

        # Calculate the standard deviation of the feature
        std_dev = np.std(feature)
        feature_std_devs.append(std_dev)

        # Plot a histogram of the feature distribution
        plt.figure(figsize=(8, 4))
        plt.hist(feature, bins=50, color='blue', alpha=0.7)
        plt.title(f'Feature {i + 1} Distribution')
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')

        # Add a text annotation with the standard deviation value
        plt.annotate(f'Std Dev: {std_dev:.2f}', xy=(0.7, 0.8), xycoords='axes fraction', fontsize=10, color='red')

        plt.show()

    return feature_std_devs
