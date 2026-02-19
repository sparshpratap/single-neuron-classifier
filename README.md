# Single Neuron Classifier

This project implements a single artificial neuron that learns to classify two-dimensional data.
The neuron is trained using a gradient-based update rule and visualized through an interactive GUI.

The project was developed as part of an Artificial Intelligence fundamentals coursework.

---

## Problem Description

The task is to implement an artificial neuron that:
- Takes two-dimensional samples as input
- Predicts class membership (0 or 1)
- Is trainable using a gradient-based learning rule

The neuron supports multiple activation functions and visualizes its decision boundary.

---

## Implemented Features

### Artificial Neuron
- Linear neuron with bias
- Trainable weights using the update rule:

Δw = η(d − y)f′(s)x

---

### Activation Functions

**Training & Evaluation**
- Heaviside (Perceptron)
- Sigmoid (Logistic)
- Sin
- Tanh

**Evaluation Only**
- Sign
- ReLU
- Leaky ReLU

---

### Data Generation
- Two-dimensional Gaussian data
- Two classes (0 and 1)
- Multiple modes per class
- Configurable sample count

---

### Visualization
- Interactive GUI using Tkinter
- Scatter plot of generated samples
- Real-time decision boundary visualization
- Activation function selection

---

## Technologies Used
- Python
- NumPy
- Matplotlib
- Tkinter (built-in)


