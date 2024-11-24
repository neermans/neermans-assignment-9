import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        if activation == 'tanh':
            self.activation_fn = np.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x) ** 2
        elif activation == 'relu':
            self.activation_fn = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.activation_fn = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: self.activation_fn(x) * (1 - self.activation_fn(x))
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_output = np.zeros((1, output_dim))

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.X = X
        self.hidden_pre_activation = X @ self.weights_input_hidden + self.bias_hidden
        self.hidden_activation = self.activation_fn(self.hidden_pre_activation)
        self.output = self.hidden_activation @ self.weights_hidden_output + self.bias_output
        return 1 / (1 + np.exp(-self.output))  # Sigmoid output for binary classification


    def backward(self, X, y):
        # Output layer gradients
        error = self.output - y
        grad_weights_hidden_output = self.hidden_activation.T @ error
        grad_bias_output = np.sum(error, axis=0, keepdims=True)

        # Hidden layer gradients
        hidden_error = error @ self.weights_hidden_output.T * self.activation_derivative(self.hidden_pre_activation)
        grad_weights_input_hidden = X.T @ hidden_error
        grad_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

        # Update weights and biases
        self.weights_hidden_output -= self.lr * grad_weights_hidden_output
        self.bias_output -= self.lr * grad_bias_output
        self.weights_input_hidden -= self.lr * grad_weights_input_hidden
        self.bias_hidden -= self.lr * grad_bias_hidden


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.hidden_activation
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Layer Activations")

    # TODO: Hyperplane visualization in the hidden space

    xx, yy = np.meshgrid(
        np.linspace(hidden_features[:, 0].min() - 1, hidden_features[:, 0].max() + 1, 50),
        np.linspace(hidden_features[:, 1].min() - 1, hidden_features[:, 1].max() + 1, 50)
    )
    grid_hidden = np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]
    decision_boundary = (grid_hidden @ mlp.weights_hidden_output.T).reshape(xx.shape)
    ax_hidden.contourf(xx, yy, decision_boundary, levels=[-1, 0, 1], cmap='bwr', alpha=0.2)


    # TODO: Distorted input space transformed by the hidden layer

    # TODO: Plot input layer decision boundary

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid_input).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=[-1, 0, 1], cmap='bwr', alpha=0.2)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title("Input Space Decision Boundary")


    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient

    gradients = np.abs(mlp.weights_input_hidden)
    for i, (x, y_val) in enumerate(zip(X[:, 0], X[:, 1])):
        circle_radius = 0.05 + 0.3 * np.mean(gradients)
        circle = Circle((x, y_val), circle_radius, color='b', alpha=0.5)
        ax_gradient.add_artist(circle)
    ax_gradient.set_xlim(x_min, x_max)
    ax_gradient.set_ylim(y_min, y_max)
    ax_gradient.set_title("Gradient Visualization")



def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)