import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Provided confusion matrix data
conf_matrix = np.array([
[747, 1, 16, 105, 131],
 [  4, 903, 4, 70, 19],
 [ 58, 0, 786, 14, 142],
 [146, 47, 5, 747, 55],
 [257, 3, 435, 74, 231]
])

# Visualizing the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=20)
plt.ylabel('True Label', fontsize=16)
plt.xlabel('Predicted Label', fontsize=16)
plt.xticks(ticks=np.arange(5) + 0.5, labels=np.arange(1, 6), fontsize=12)
plt.yticks(ticks=np.arange(5) + 0.5, labels=np.arange(1, 6), fontsize=12, rotation=0)
plt.show()
