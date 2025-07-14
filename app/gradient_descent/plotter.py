import matplotlib.pyplot as plt
import numpy as np


def plot_cost_vs_w(P_history, J_history):
    w_history = [p[0] for p in P_history]
    plt.plot(w_history, J_history, marker="*", linestyle="-", color="red")

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
    ax1.plot(J_history[:after_iteration])
    ax2.plot(
        after_iteration + np.arange(len(J_history[after_iteration:])),
        J_history[after_iteration:],
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


def plot_placement_data(data):
    colors = ["red" if status == 0 else "green" for status in data["placed"]]
    plt.figure(figsize=(8, 6))
    plt.scatter(data["cgpa"], data["placement_exam_marks"], c=colors, edgecolors="k")
    plt.xlabel("GPA")
    plt.ylabel("Placement Mark")
    plt.title("Placement Status by GPA and Placement Mark")
    plt.grid(True)
    plt.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Not Placed",
                markerfacecolor="red",
                markersize=8,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Placed",
                markerfacecolor="green",
                markersize=8,
            ),
        ]
    )
    plt.show()


def plot_decision_boundary(X, y, w, b, title="Decision Boundary"):
    colors = ["red" if label == 0 else "green" for label in y]

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="k")
    plt.xlabel("Scaled GPA")
    plt.ylabel("Scaled Placement Marks")
    plt.title(title)
    plt.grid(True)

    # Create decision boundary: where sigmoid(w·x + b) = 0.5 => w·x + b = 0
    x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, label="Decision Boundary", color="blue")
    else:
        # Vertical line if w[1] = 0
        x_val = -b / w[0]
        plt.axvline(x=x_val, color="blue", label="Decision Boundary")

    plt.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Not Placed",
                markerfacecolor="red",
                markersize=8,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Placed",
                markerfacecolor="green",
                markersize=8,
            ),
            plt.Line2D([0], [0], color="blue", label="Decision Boundary"),
        ]
    )

    plt.show()
