from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def run_benchmark(
    X_train, y_train, w_predicted, b_predicted, X_test, y_test, y_pred_custom
):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nRunning comparisions ...... \n")
    print(f"My model b:{ b_predicted}, W:{w_predicted}")
    print(f"Scikit model b:{ model.intercept_}, W:{model.coef_}")
    y_test_standard = model.predict(X_test)
    compute_performance(
        y_test=y_test, y_pred_custom=y_pred_custom, y_pred_standard=y_test_standard
    )

    if X_train.shape[1] == 1:
        visualize_comparisions(
            X_test=X_test,
            y_test=y_test,
            y_pred_custom=y_pred_custom,
            y_pred_standard=y_test_standard,
        )


def compute_performance(y_test, y_pred_custom, y_pred_standard):
    print("MSE (Custom):", mean_squared_error(y_test, y_pred_custom))
    print("MSE (Standard):", mean_squared_error(y_test, y_pred_standard))

    print("R² (Custom):", r2_score(y_test, y_pred_custom))
    print("R² (Standard):", r2_score(y_test, y_pred_standard))


def visualize_comparisions(X_test, y_test, y_pred_custom, y_pred_standard):
    plt.scatter(X_test, y_test, color="black", label="True values")
    plt.plot(X_test, y_pred_custom, color="red", label="Custom model", marker="o")
    plt.plot(X_test, y_pred_standard, color="blue", label="Sklearn model", marker="x")
    plt.legend()
    plt.show()
