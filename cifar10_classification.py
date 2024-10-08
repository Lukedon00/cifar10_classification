# CIFAR-10 Classification using Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN)

# Import necessary libraries
import tensorflow as tf # type: ignore
from tensorflow.keras import datasets, layers, models # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import ssl

# To avoid SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Reshape the labels from 2D to 1D
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# Class labels
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to plot sample images
def plot_sample(x, y, index):
    plt.figure(figsize=(6,2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])

# Plot a sample image
plot_sample(x_train, y_train, 1854)

# Normalize the data (scaling pixel values to the range 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build a simple ANN model
ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(30, activation='relu'),   # First hidden layer with ReLU
    layers.Dense(15, activation='relu'),   # Second hidden layer with ReLU
    layers.Dense(10, activation='sigmoid') # Output layer with Sigmoid activation
])

# Compile the ANN model
ann.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Train the ANN model
ann.fit(x_train, y_train, epochs=10)

# Evaluate the ANN model on the test set
ann.evaluate(x_test, y_test)

# Make predictions on the test set
y_prediction = ann.predict(x_test)
y_pred = [np.argmax(i) for i in y_prediction]

# Print classification report for ANN model
print('Classification report (ANN): \n', classification_report(y_test, y_pred))

# Confusion matrix for ANN model
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)
print('Confusion Matrix (ANN):\n', cm)

# Build a CNN model
cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(200, activation='relu'),   # First hidden layer with ReLU
    layers.Dense(100, activation='relu'),   # Second hidden layer with ReLU
    layers.Dense(10, activation='softmax')  # Output layer with Softmax activation for multi-class classification
])

# Compile the CNN model
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Train the CNN model
cnn.fit(x_train, y_train, epochs=10)

# Evaluate the CNN model on the test set
cnn.evaluate(x_test, y_test)

# Make predictions on the test set
y_pred = cnn.predict(x_test)
y_classes = [np.argmax(element) for element in y_pred]

# Print classification report for CNN model
print('Classification report (CNN): \n', classification_report(y_test, y_classes))
