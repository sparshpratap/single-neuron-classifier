import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from neuron import Neuron
from activations import (
    heaviside, heaviside_derivative,
    sigmoid, sigmoid_derivative,
    sin_activation, sin_derivative,
    tanh_activation, tanh_derivative,
    sign_activation, relu, leaky_relu
) 
from data_generator import generate_gaussian_data


class NeuronGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Single Neuron Classifier")

        self.neuron = Neuron()
        self.X = None
        self.y = None

        self._build_controls()
        self._build_plot()

    # ---------------- GUI LAYOUT ----------------

    def _build_controls(self):
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, padx=10)

        tk.Label(frame, text="Activation function").pack()

        self.activation_var = tk.StringVar(value="Heaviside")
        activations = [
            "Heaviside", "Sigmoid", "Sin", "Tanh",
            "Sign", "ReLU", "Leaky ReLU"
        ]

        ttk.OptionMenu(frame, self.activation_var, activations[0], *activations).pack()

        tk.Button(frame, text="Generate data", command=self.generate_data).pack(pady=5)
        tk.Button(frame, text="Train neuron", command=self.train_neuron).pack(pady=5)

    def _build_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT)

    # ---------------- LOGIC ----------------

    def generate_data(self):
        self.X, self.y = generate_gaussian_data(
            modes_per_class=2,
            samples_per_mode=50
        )
        self.plot()

    def train_neuron(self):
        if self.X is None:
            return

        act, d_act = self.get_activation_functions()
        if d_act is None:
            return

        self.neuron.train(self.X, self.y, act, d_act, epochs=50)
        self.plot()

    def get_activation_functions(self):
        name = self.activation_var.get()

        if name == "Heaviside":
            return heaviside, heaviside_derivative
        if name == "Sigmoid":
            return sigmoid, sigmoid_derivative
        if name == "Sin":
            return sin_activation, sin_derivative
        if name == "Tanh":
            return tanh_activation, tanh_derivative

        # Evaluation only (no training)
        if name == "Sign":
            return sign_activation, None
        if name == "ReLU":
            return relu, None
        if name == "Leaky ReLU":
            return leaky_relu, None

    # ---------------- PLOTTING ----------------

    def plot(self):
        self.ax.clear()

        if self.X is not None:
            self.ax.scatter(self.X[self.y == 0][:, 0],
                            self.X[self.y == 0][:, 1],
                            color="blue", label="Class 0")

            self.ax.scatter(self.X[self.y == 1][:, 0],
                            self.X[self.y == 1][:, 1],
                            color="red", label="Class 1")

            self.plot_decision_boundary()

        self.ax.legend()
        self.canvas.draw()

    def plot_decision_boundary(self):
        w = self.neuron.w
        b = self.neuron.b

        x_vals = np.linspace(-2, 2, 200)
        y_vals = np.linspace(-2, 2, 200)
        xx, yy = np.meshgrid(x_vals, y_vals)

        grid = np.c_[xx.ravel(), yy.ravel()]
        z = np.dot(grid, w) + b
        z = z.reshape(xx.shape)

        self.ax.contourf(xx, yy, z >= 0, alpha=0.2, colors=["blue", "red"])


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuronGUI(root)
    root.mainloop()
