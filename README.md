# Perceptron
A Perceptron is a fundamental unit of a neural network that models a single-layer binary classifier. It takes weighted inputs, applies an activation function (often a step function), and produces an output. The perceptron learns by adjusting weights based on errors, using an iterative learning algorithm.

Overview

The perceptron is the simplest type of artificial neural network, used for binary classification tasks. It is a single-layer model that applies a weighted sum of inputs followed by an activation function to determine the output. It is the foundation of more complex neural networks.

Key Features:
1.Linear Classifier: Suitable for linearly separable data.
2.Supervised Learning: Trains using labeled datasets.
3.Threshold Activation: Uses a step function to determine output.
4. Adjustable Weights: Learns by updating weights using the perceptron learning rule.

Algorithm:

1. Initialize weights and bias.
2. Compute weighted sum: y = W*X + b
3. Apply activation function (e.g., step function).
4. Update weights using the Perceptron Learning Rule:
5. If the prediction is incorrect, update: W = W + learning_rate * (y_actual - y_predicted) * X
6. Repeat until convergence.

Implementation Example (Python):
import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_train(X, y, lr=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0
    for _ in range(epochs):
        for i in range(len(X)):
            prediction = step_function(np.dot(X[i], weights) + bias)
            weights += lr * (y[i] - prediction) * X[i]
            bias += lr * (y[i] - prediction)
    return weights, bias

Applications:
1.Binary classification (e.g., spam detection, sentiment analysis)
2.Pattern recognition
3.Basic neural network foundations

Limitations:
1. Can only classify linearly separable data.
2. Does not handle complex decision boundaries.
