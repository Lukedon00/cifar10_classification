# CIFAR-10 Classification with ANN and CNN

This project demonstrates the classification of the CIFAR-10 dataset (images of 10 different object classes) using both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) built with Keras and TensorFlow.

## Project Overview

- **Dataset**: CIFAR-10 dataset from `tensorflow.keras.datasets`
- **Algorithm**: Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN)
- **Objective**: Classify objects such as airplanes, automobiles, birds, etc., based on pixel values.
- **Activation Functions**: Sigmoid, ReLU, Softmax
- **Tools Used**: Python, Keras, TensorFlow, Matplotlib, Numpy, Seaborn

## Steps:
1. Load and preprocess the CIFAR-10 dataset.
2. Build a simple ANN model and train it for 10 epochs.
3. Build a CNN model with convolutional layers and train it for 10 epochs.
4. Evaluate both models using accuracy, confusion matrix, and classification report.

## Results:
- **Accuracy (ANN)**: Achieved moderate accuracy with a simple ANN model.
- **Accuracy (CNN)**: Achieved higher accuracy with a deeper CNN model.
- **Confusion Matrix**: Visualized for both ANN and CNN models.
- **Classification Report**: Provides precision, recall, and F1-score for each class.

## Why Sigmoid, ReLU, and Softmax?
- **Sigmoid**: Used in the output layer of the simple ANN model to squash the values between 0 and 1 for binary-like output.
- **ReLU**: Used in the hidden layers of both the ANN and CNN models to avoid vanishing gradient problems and to speed up training.
- **Softmax**: Used in the output layer of the CNN model to convert the raw outputs into probabilities for multi-class classification.

## How to Run

### Prerequisites
- Python 3.x
- Required Libraries (see `requirements.txt`)

### Steps to Run:
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/cifar10_classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd cifar10_classification
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook or the Python script:
    - Jupyter Notebook: Open the `cifar10_classification.ipynb` file and run the cells.
    - Python Script: Execute the Python file in your terminal:
    ```bash
    python cifar10_classification.py
    ```

## License
This project is licensed under the MIT License.
