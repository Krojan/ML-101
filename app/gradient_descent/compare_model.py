from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
    mean_squared_error,
    r2_score,
)


def run_benchmark(
    X_train,
    y_train,
    w_predicted,
    b_predicted,
    X_test,
    y_test,
    y_pred_custom,
    linear=True,
):
    if linear:
        model = LinearRegression()
    else:
        model = LogisticRegression()

    model.fit(X_train, y_train)
    print("\nRunning comparisions ...... \n")
    print(f"My model b:{ b_predicted}, W:{w_predicted}")
    print(f"Scikit model b:{ model.intercept_}, W:{model.coef_}")
    y_test_standard = model.predict(X_test)

    compute_performance(
        linear=linear,
        y_test=y_test,
        y_pred_custom=y_pred_custom,
        y_pred_standard=y_test_standard,
    )

    if X_train.shape[1] == 1:
        visualize_comparisions(
            X_test=X_test,
            y_test=y_test,
            y_pred_custom=y_pred_custom,
            y_pred_standard=y_test_standard,
        )

    if not linear:
        visualize_classification_comparisons(
            X=X_test,
            y_true=y_test,
            y_pred_custom=y_pred_custom,
            y_pred_standard=y_test_standard,
        )


def compute_performance(y_test, y_pred_custom, y_pred_standard, linear):
    if linear:
        print("MSE (Custom):", mean_squared_error(y_test, y_pred_custom))
        print("MSE (Standard):", mean_squared_error(y_test, y_pred_standard))

        print("R² (Custom):", r2_score(y_test, y_pred_custom))
        print("R² (Standard):", r2_score(y_test, y_pred_standard))
    else:
        # compute erros
        print("\n--- Logistic Regression Performance ---")

        print("Accuracy (Custom):", accuracy_score(y_test, y_pred_custom))
        print("Accuracy (Standard):", accuracy_score(y_test, y_pred_standard))

        print("F1 Score (Custom):", f1_score(y_test, y_pred_custom))
        print("F1 Score (Standard):", f1_score(y_test, y_pred_standard))

        print("Precision (Custom):", precision_score(y_test, y_pred_custom))
        print("Precision (Standard):", precision_score(y_test, y_pred_standard))

        print("Recall (Custom):", recall_score(y_test, y_pred_custom))
        print("Recall (Standard):", recall_score(y_test, y_pred_standard))

        print("\nConfusion Matrix (Custom):\n", confusion_matrix(y_test, y_pred_custom))
        print(
            "Confusion Matrix (Standard):\n", confusion_matrix(y_test, y_pred_standard)
        )

        # Optional: Full report
        print(
            "\nClassification Report (Custom):\n",
            classification_report(y_test, y_pred_custom),
        )
        print(
            "Classification Report (Standard):\n",
            classification_report(y_test, y_pred_standard),
        )


def visualize_comparisions(X_test, y_test, y_pred_custom, y_pred_standard):
    plt.scatter(X_test, y_test, color="black", label="True values")
    plt.plot(X_test, y_pred_custom, color="red", label="Custom model", marker="o")
    plt.plot(X_test, y_pred_standard, color="blue", label="Sklearn model", marker="x")
    plt.legend()
    plt.show()


def visualize_classification_comparisons(
    X, y_true, y_pred_custom, y_pred_standard, title="Model Comparison"
):
    plt.figure(figsize=(8, 6))

    # Map true labels to color
    colors = ["red" if label == 0 else "green" for label in y_true]

    # Plot true data points
    # plt.scatter(
    #     X[:, 0], X[:, 1], c=colors, edgecolors="k", label="True labels", alpha=0.6
    # )

    plt.scatter(
        X[y_true == 0, 0],
        X[y_true == 0, 1],
        c="red",
        edgecolors="k",
        label="Label 0",
        alpha=0.6,
    )
    plt.scatter(
        X[y_true == 1, 0],
        X[y_true == 1, 1],
        c="green",
        edgecolors="k",
        label="Label 1",
        alpha=0.6,
    )

    # Mark mismatches
    # Overlay custom model predictions (contour or markers)
    custom_correct = y_pred_custom == y_true
    standard_correct = y_pred_standard == y_true

    # Mark mismatches in custom model
    plt.scatter(
        X[~custom_correct][:, 0],
        X[~custom_correct][:, 1],
        marker="x",
        color="red",
        label="Custom Wrong",
        s=80,
    )

    # Mark mismatches in sklearn model
    plt.scatter(
        X[~standard_correct][:, 0],
        X[~standard_correct][:, 1],
        marker="o",
        facecolors="none",
        edgecolors="blue",
        label="Sklearn Wrong",
        s=80,
    )

    plt.xlabel("Scaled GPA")
    plt.ylabel("Scaled Placement Marks")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
