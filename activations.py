import numpy as np

# ===== Heaviside =====
def heaviside(s):
    return 1 if s >= 0 else 0

def heaviside_derivative(s):
    # Assumed to be 1 
    return 1


# ===== Logistic (Sigmoid) =====
def sigmoid(s, beta=1.0):
    return 1 / (1 + np.exp(-beta * s))

def sigmoid_derivative(s, beta=1.0):
    y = sigmoid(s, beta)
    return beta * y * (1 - y)


# ===== Sin =====
def sin_activation(s):
    return np.sin(s)

def sin_derivative(s):
    return np.cos(s)


# ===== Tanh =====
def tanh_activation(s):
    return np.tanh(s)

def tanh_derivative(s):
    return 1 - np.tanh(s) ** 2


# ===== Sign (evaluation only) =====
def sign_activation(s):
    if s < 0:
        return -1
    elif s == 0:
        return 0
    else:
        return 1


# ===== ReLU (evaluation only) =====
def relu(s):
    return max(0, s)


# ===== Leaky ReLU (evaluation only) =====
def leaky_relu(s, alpha=0.01):
    return s if s > 0 else alpha * s
