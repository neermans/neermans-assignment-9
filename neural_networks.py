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
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        
        # Initialize weights and biases 
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))
        
        # Variables to store activations and gradients
        self.hidden_activations = None
        self.dW1 = None
        self.dW2 = None
        self.output = None
        

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        if self.activation_fn == 'tanh':
            self.A1 = np.tanh(self.Z1)
        elif self.activation_fn == 'sigmoid':
            self.A1 = 1 / (1 + np.exp(-self.Z1))
        elif self.activation_fn == 'relu':
            self.A1 = np.maximum(0, self.Z1)
        
        # Store hidden activations for visualization
        self.hidden_activations = self.A1
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        
        # For binary classification, we use sigmoid activation at the output layer
        out = 1 / (1 + np.exp(-self.Z2))
        self.output = out  # Store output for backward pass
        
        return out

    def backward(self, X, y):
        m = y.shape[0]
        # Compute gradients
        Z2_der = self.output - y  # Derivative of loss w.r.t Z2
        W2_der = np.dot(self.A1.T, Z2_der) / m
        db2 = np.sum(Z2_der, axis=0, keepdims=True) / m
        
        dA1 = np.dot(Z2_der, self.W2.T)
        if self.activation_fn == 'tanh':
            dZ1 = dA1 * (1 - self.A1 ** 2)
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (self.Z1 > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            dZ1 = dA1 * self.A1 * (1 - self.A1)

        W1_der = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights with gradient descent
        self.W1 -= self.lr * W1_der
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * W2_der
        self.b2 -= self.lr * db2
        
        # Store gradients for visualization
        self.dW1 = W1_der
        self.dW2 = W2_der

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward functions
    for _ in range(10):
        out = mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.hidden_activations
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")

    # Set axis limits based on activation function
    if mlp.activation_fn == 'tanh':
        ax_hidden.set_xlim([-1, 1])
        ax_hidden.set_ylim([-1, 1])
        ax_hidden.set_zlim([-1, 1])
    elif mlp.activation_fn == 'sigmoid':
        ax_hidden.set_xlim([0, 1])
        ax_hidden.set_ylim([0, 1])
        ax_hidden.set_zlim([0, 1])
    elif mlp.activation_fn == 'relu':
        max_val = np.max(mlp.hidden_activations)
        min_val = np.min(mlp.hidden_activations)
        buffer = 0.1 * (max_val - min_val)  # Add some buffer space
        ax_hidden.set_xlim([min_val - buffer, max_val + buffer])
        ax_hidden.set_ylim([min_val - buffer, max_val + buffer])
        ax_hidden.set_zlim([min_val - buffer, max_val + buffer])

    # Distorted input space transformed by the hidden layer
    grid_range = np.linspace(-3, 3, 20)
    X_grid, Y_grid = np.meshgrid(grid_range, grid_range)
    grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]

    # Transform grid points through the hidden layer
    Z1_grid = np.dot(grid_points, mlp.W1) + mlp.b1
    if mlp.activation_fn == 'tanh':
        A1_grid = np.tanh(Z1_grid)
    elif mlp.activation_fn == 'relu':
        A1_grid = np.maximum(0, Z1_grid)
    elif mlp.activation_fn == 'sigmoid':
        A1_grid = 1 / (1 + np.exp(-Z1_grid))
    else:
        raise ValueError("Unsupported activation function")

    # Reshape for plotting
    H1 = A1_grid[:, 0].reshape(X_grid.shape)
    H2 = A1_grid[:, 1].reshape(Y_grid.shape)
    H3 = A1_grid[:, 2].reshape(Y_grid.shape)

    # Plot the distorted plane in the hidden space
    ax_hidden.plot_surface(H1, H2, H3, alpha=0.25, color='lightgrey', edgecolor='none')

    # Hyperplane visualization in the hidden space
    w20, w21, w22 = mlp.W2[:, 0]
    b2 = mlp.b2[0, 0]

    # Adjust h_range based on activation function
    if mlp.activation_fn == 'tanh':
        h_range = np.linspace(-1, 1, 10)  # Fixed range for tanh
    elif mlp.activation_fn == 'sigmoid':
        h_range = np.linspace(0, 1, 10)  # Fixed range for sigmoid
    elif mlp.activation_fn == 'relu':
        max_val = np.max(mlp.hidden_activations)
        min_val = np.min(mlp.hidden_activations)
        buffer = 0.1 * (max_val - min_val) if max_val > min_val else 0.5
        h_range = np.linspace(min_val - buffer, max_val + buffer, 10)  # Expanding range for ReLU

    H1_plane, H2_plane = np.meshgrid(h_range, h_range)

    if w22 != 0:
        H3_plane = (-w20 * H1_plane - w21 * H2_plane - b2) / w22
        ax_hidden.plot_surface(H1_plane, H2_plane, H3_plane, alpha=0.5, color='orange')

    # Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.forward(grid_input)
    probs = probs.reshape(xx.shape)
    ax_input.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title(f"Input Space at Step {frame * 10}")

    # Visualize features and gradients as circles and edges
    neuron_positions = {
        'input': [(0, 0.1), (0, 0.9)],
        'hidden': [(0.5, 0.2), (0.5, 0.5), (0.5, 0.8)],
        'output': [(1, 0.5)],
    }
    neuron_labels = {
        'input': ['x1', 'x2'],
        'hidden': ['h1', 'h2', 'h3'],
        'output': ['y'],
    }
    # Plot neurons and add labels
    for idx, pos in enumerate(neuron_positions['input']):
        circle = Circle(pos, 0.05, color='blue')
        ax_gradient.add_patch(circle)
        ax_gradient.text(pos[0], pos[1] + 0.1, neuron_labels['input'][idx],
                         ha='center', va='bottom', fontsize=12)
    for idx, pos in enumerate(neuron_positions['hidden']):
        circle = Circle(pos, 0.05, color='green')
        ax_gradient.add_patch(circle)
        ax_gradient.text(pos[0], pos[1] + 0.1, neuron_labels['hidden'][idx],
                         ha='center', va='bottom', fontsize=12)
    for idx, pos in enumerate(neuron_positions['output']):
        circle = Circle(pos, 0.05, color='red')
        ax_gradient.add_patch(circle)
        ax_gradient.text(pos[0], pos[1] + 0.1, neuron_labels['output'][idx],
                         ha='center', va='bottom', fontsize=12)
    ax_gradient.set_title(f"Gradients at Step {frame * 10}", y=1.05)

    # Plot edges with thickness and color based on gradient values
    scaling_factor = 100  # Adjust as needed for visibility

    # From input to hidden
    for i, input_pos in enumerate(neuron_positions['input']):
        for j, hidden_pos in enumerate(neuron_positions['hidden']):
            grad = mlp.dW1[i, j]
            if grad < 0:
                line_width = min(-grad * scaling_factor, 100)  # Thick for negative gradients
            else:
                line_width = min(grad * scaling_factor, 100)  # Thin for positive gradients
            ax_gradient.plot([input_pos[0], hidden_pos[0]],
                            [input_pos[1], hidden_pos[1]],
                            'k-', lw=line_width)

    # From hidden to output
    for i, hidden_pos in enumerate(neuron_positions['hidden']):
        for output_pos in neuron_positions['output']:
            grad = mlp.dW2[i, 0]
            if grad < 0:
                line_width = min(-grad * scaling_factor, 100)  # Thick for negative gradients
            else:
                line_width = min(grad * scaling_factor, 100)  # Thin for positive gradients
            ax_gradient.plot([hidden_pos[0], output_pos[0]],
                            [hidden_pos[1], output_pos[1]],
                            'k-', lw=line_width)

    ax_gradient.set_xlim(-0.1, 1.1)
    ax_gradient.set_ylim(-0.1, 1.1)
    ax_gradient.axis('off')
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")


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
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input,
                                     ax_hidden=ax_hidden, ax_gradient=ax_gradient,
                                     X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1  # Adjust if necessary
    step_num = 1000
    visualize(activation, lr, step_num)