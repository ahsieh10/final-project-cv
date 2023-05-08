import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('./training_accuracies.csv')

# Extract the metrics data
sparse_categorical_accuracy = df['sparse_categorical_accuracy']
val_sparse_categorical_accuracy = df['val_sparse_categorical_accuracy']

# Plot the metrics
epochs = range(1, len(sparse_categorical_accuracy) + 1)
plt.plot(epochs, sparse_categorical_accuracy, 'm', label='Training Sparse Categorical Accuracy')
plt.plot(epochs, val_sparse_categorical_accuracy, 'b', label='Validation Sparse Categorical Accuracy')
plt.title('Training and Validation Sparse Categorical Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Sparse Categorical Accuracy')
plt.legend()
plt.show()
