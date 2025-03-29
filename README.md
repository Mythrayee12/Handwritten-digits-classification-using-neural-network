### Handwritten Digits Classification using Neural Network

# Overview

This project implements a Handwritten Digits Classification system using a Neural Network. The model is trained on the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The neural network is built using TensorFlow/Keras and achieves high accuracy in recognizing digits from handwritten images.

![image](https://github.com/user-attachments/assets/a9b86d8c-91a5-4b73-9191-ba365b0d2297)

# Features

Multi-layer neural network for digit classification

Uses MNIST dataset for training and testing

Implements ReLU activation and Softmax classifier

Adam optimizer for efficient training

Visualizes training progress and performance metrics



# Tech Stack

Python

TensorFlow/Keras

NumPy, Matplotlib, Seaborn (for data visualization)

Dataset

The MNIST dataset contains:

60,000 training images

10,000 testing images

You can load the dataset using:

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Model Architecture

The neural network consists of:

Input layer (28x28 flattened to 784 neurons)

Hidden layers (Fully connected layers with ReLU activation)

Output layer (10 neurons with Softmax activation)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])



# Training

Compile and train the model:

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))



# Evaluation

Evaluate the model on the test dataset:

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")



# Sample Predictions

Plot sample test images with model predictions:

import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(model, x_test, y_test):
    predictions = model.predict(x_test[:10])
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f'Predicted: {np.argmax(predictions[i])}')
        plt.axis('off')
    plt.show()

plot_predictions(model, x_test, y_test)



# How to Run

Clone the repository:


git clone https://github.com/your-username/handwritten-digits-classification.git
cd handwritten-digits-classification



 Install dependencies:


pip install tensorflow numpy matplotlib seaborn

Run the training script:

python train.py

View model predictions:

python predict.py


