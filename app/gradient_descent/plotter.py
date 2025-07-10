import matplotlib.pyplot as plt
import numpy as np


def plot_cost_vs_w(P_history, J_history):
    w_history = [p[0] for p in P_history]
    plt.plot(w_history, J_history, marker="*", linestyle="-")

    # Labels and title
    plt.xlabel("Value of w")
    plt.ylabel("Cost (J)")
    plt.title("Cost vs. Value of w")

    # Show the plot
    plt.show()


def plot_cost_vs_b(P_history, J_history):
    b_history = [p[1] for p in P_history]

    plt.plot(b_history, J_history, marker="*", linestyle="-")

    # Labels and title
    plt.xlabel("Value of b")
    plt.ylabel("Cost (J)")
    plt.title("Cost vs. Value of b")

    # Show the plot
    plt.show()


def plot_cost_vs_iteration(J_history, after_iteration=100):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_history[:100])
    ax2.plot(
        1000 + np.arange(len(J_history[after_iteration:])), J_history[after_iteration:]
    )
    ax1.set_title("Cost vs. iteration(start)")
    ax2.set_title("Cost vs. iteration (end)")
    ax1.set_ylabel("Cost")
    ax2.set_ylabel("Cost")
    ax1.set_xlabel("iteration step")
    ax2.set_xlabel("iteration step")
    plt.show()


def plot_linear_regression(X, Y, w, b):
    # Assume these are already defined:
    # X: training inputs (1D or reshaped 2D)
    # Y: training outputs
    # w: learned weight (slope)
    # b: learned bias (intercept)

    # Ensure X is a numpy array
    X = np.array(X)
    Y = np.array(Y)

    # Predict Y values using your model
    Y_pred = w * X + b  # or use your model.predict(X)

    # Plot training data
    plt.scatter(X, Y, color="blue", label="Training Data")

    # Plot the regression line
    plt.plot(X, Y_pred, color="red", label="Fitted Line")

    # Add labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_data(X, y, title, color, xlabel="X", ylabel="Y"):
    plt.scatter(X, y, color=color, label=title)
    # plt.scatter
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_multiple_datasets(datasets):
    for dataset in datasets:
        plt.scatter(
            dataset["X"], dataset["y"], color=dataset["color"], label=dataset["title"]
        )
    plt.legend()
    plt.show()
