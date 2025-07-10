from linear_regression import linear_regression
from load_data import *
from logistic_regression import logistic_regression
import plotter as plt
import os
from compare_model import run_benchmark

current_dir = os.path.dirname(os.path.abspath(__file__))


def run_linear_regression_multiple_features():
    feature = "multiple"
    X_train, y_train, w, b, _, alpha, iters, X_test, y_test = load_linear_data(feature)
    linear_reg = linear_regression(X=X_train, y=y_train, w_init=w, b_init=b)
    w_final, b_final, J_history, P_history = linear_reg.train(
        alpha=alpha, num_iterations=iters
    )
    y_test_predictions = linear_reg.predict(X_train, y_train)
    plt.plot_cost_vs_iteration(J_history)
    run_benchmark(
        X_train=X_train,
        y_train=y_train,
        w_predicted=w_final,
        b_predicted=b_final,
        X_test=X_test,
        y_test=y_test,
        y_pred_custom=y_test_predictions,
    )


def run_linear_regression_single_feature():
    feature = "single"

    X_train, y_train, w, b, _, alpha, iters, X_test, y_test = load_linear_data(feature)
    linear_reg = linear_regression(X=X_train, y=y_train, w_init=w, b_init=b)
    w_final, b_final, J_history, P_history = linear_reg.train(
        alpha=alpha, num_iterations=iters
    )
    y_test_predictions = linear_reg.predict(X_train, y_train)
    plt.plot_cost_vs_iteration(J_history)
    plt.plot_cost_vs_w(P_history, J_history)
    plt.plot_cost_vs_b(P_history, J_history)
    plt.plot_linear_regression(X_train, y_train, w_final, b_final)

    run_benchmark(
        X_train=X_train,
        y_train=y_train,
        w_predicted=w_final,
        b_predicted=b_final,
        X_test=X_test,
        y_test=y_test,
        y_pred_custom=y_test_predictions,
    )


def run_linear_regression_on_salary_prediction():
    relative_filepath = "../../sample/Files/salary.csv"
    filepath = os.path.join(current_dir, relative_filepath)
    filepath = os.path.abspath(filepath)
    (
        X,
        Y,
        X_train,
        y_train,
        w,
        b,
        alpha,
        iters,
        X_test,
        y_test,
    ) = load_salary_data(filepath)
    linear_reg = linear_regression(X=X_train, y=y_train, w_init=w, b_init=b)
    w_final, b_final, J_history, P_history = linear_reg.train(
        alpha=alpha, num_iterations=iters, regularization=True
    )
    y_test_predictions = linear_reg.predict(X_test, y_test)
    plt.plot_cost_vs_iteration(J_history)
    plt.plot_cost_vs_w(P_history, J_history)
    plt.plot_cost_vs_b(P_history, J_history)
    plt.plot_linear_regression(X, Y, w_final, b_final)

    run_benchmark(
        X_train=X_train,
        y_train=y_train,
        w_predicted=w_final,
        b_predicted=b_final,
        X_test=X_test,
        y_test=y_test,
        y_pred_custom=y_test_predictions,
    )


def run_linear_regression_on_gpa_prediction():
    # predict gpa based on sat score
    relative_filepath = "../../sample/Files/sat_gpa_score.csv"
    filepath = os.path.join(current_dir, relative_filepath)
    filepath = os.path.abspath(filepath)
    (
        X,
        Y,
        X_train,
        y_train,
        w,
        b,
        alpha,
        iters,
        X_test,
        y_test,
    ) = load_gpa_data(filepath)

    linear_reg = linear_regression(X=X_train, y=y_train, w_init=w, b_init=b)

    w_final, b_final, J_history, P_history = linear_reg.train(
        alpha=alpha, num_iterations=iters
    )

    y_test_predictions = linear_reg.predict(X_test, y_test)
    plt.plot_cost_vs_iteration(J_history)
    plt.plot_cost_vs_w(P_history, J_history)
    plt.plot_cost_vs_b(P_history, J_history)
    plt.plot_linear_regression(X, Y, w_final, b_final)

    run_benchmark(
        X_train=X_train,
        y_train=y_train,
        w_predicted=w_final,
        b_predicted=b_final,
        X_test=X_test,
        y_test=y_test,
        y_pred_custom=y_test_predictions,
    )


def run_logistic_regression():
    relative_filepath = "./../../sample/Files/ex2data1.txt"
    filepath = os.path.join(current_dir, relative_filepath)
    filepath = os.path.abspath(filepath)
    X_train, y_train, w, b, lamda, alpha, iters, _, _ = load_logistic_data(filepath)
    logistic_reg = logistic_regression(X=X_train, y=y_train, w_init=w, b_init=b)

    # cost = logistic_reg.compute_cost()
    # print(f"logistic reg cost = {cost}")

    logistic_reg.train(alpha=alpha, num_iterations=iters)
    _ = logistic_reg.predict(X_train, y_train)


def run_logistic_regression_on_placement_data():
    pass


def run_regularization():
    X_train, y_train, w_init, b_init, lamda = load_regularization_data()

    print(f"\nRunning regularization for linear regressionn .....")
    linear_reg = linear_regression(X=X_train, y=y_train, w_init=w_init, b_init=b_init)
    linear_reg.train(alpha=0, num_iterations=1, regularization=True, lamda=lamda)

    print(f"\nRunning regularization for logistic regression ....")

    logistic_reg = logistic_regression(
        X=X_train, y=y_train, w_init=w_init, b_init=b_init
    )
    logistic_reg.train(alpha=0, num_iterations=1, regularization=True, lamda=lamda)


def main(choice):
    if choice == "linear-multiple":
        run_linear_regression_multiple_features()
    if choice == "linear-single":
        run_linear_regression_single_feature()
    elif choice == "salary":
        run_linear_regression_on_salary_prediction()
    elif choice == "logistic":
        run_logistic_regression()
    elif choice == "regular":
        run_regularization()
    elif choice == "gpa":
        run_linear_regression_on_gpa_prediction()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print(
            "Usage: python index.py [linear-single|linear-multiple|logistic|regular|salary]"
        )
