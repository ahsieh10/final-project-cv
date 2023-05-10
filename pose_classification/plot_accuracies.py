import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('./csv_data/training_accuracies.csv')

# Extract the data
sparse_categorical_accuracy = df['sparse_categorical_accuracy']
val_sparse_categorical_accuracy = df['val_sparse_categorical_accuracy']

# Plot the accuracy data
epochs = range(1, len(sparse_categorical_accuracy) + 1)
plt.plot(epochs, sparse_categorical_accuracy, 'm', label='Training Sparse Categorical Accuracy')
plt.plot(epochs, val_sparse_categorical_accuracy, 'b', label='Validation Sparse Categorical Accuracy')
plt.title('Pose Detection Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Sparse Categorical Accuracy')
plt.legend()
plt.show()
