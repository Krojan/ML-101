import numpy as np
import pandas as pd
from plotter import plot_data, plot_placement_data
from sklearn.preprocessing import StandardScaler


# Step 4 : train test split
from sklearn.model_selection import train_test_split


def load_linear_data(feature="single"):
    if feature == "single":
        iters = 100000
        alpha = 0.01
        X_train = np.array([[1], [2]])
        y_train = np.array([300, 500])
    if feature == "multiple":
        alpha = 5.0e-7
        iters = 1000
        X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
        y_train = np.array([460, 232, 178])

    w = np.zeros(X_train[0].shape[0])
    b = 0
    lamda = 0.07
    return X_train, y_train, w, b, lamda, alpha, iters, X_train, y_train


def load_regularization_data():
    np.random.seed(1)
    X_train = np.random.rand(5, 3)
    y_train = np.array([0, 1, 0, 1, 0])
    w_init = (
        np.random.rand(X_train.shape[1])
        # .reshape(
        #     -1,
        # )
        # - 0.5
    )
    b_init = 0.5
    lamda = 0.7
    return X_train, y_train, w_init, b_init, lamda


def load_logistic_data(filename):
    # # Initialize fitting parameters
    # np.random.seed(1)
    # initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    # initial_b = 1.0

    # Set regularization parameter lambda_ (you can try varying this)
    lamda = 0.01

    # Some gradient descent settings
    iterations = 10000
    alpha = 0.001
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :2]

    y = data[:, 2]
    # Compute and display cost with w and b initialized to zeros
    initial_w = np.array([-0.00353244, -0.00407661])
    initial_b = -8

    # initial_w = np.array([0.2, -0.5])
    # initial_b = -24.0

    return X, y, initial_w, initial_b, lamda, alpha, iterations, X, y


def load_salary_data(filename):
    salary = pd.read_csv(filename)
    print("Importing from csv, columnns= ", salary.columns)
    y = np.array(salary["Salary"])
    X = np.array(salary[["Experience Years"]])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=2529
    )
    w = np.zeros(X_train[0].shape[0])
    b = 0.09
    alpha = 0.005
    iters = 100000
    return (
        X,
        y,
        X_train,
        y_train,
        w,
        b,
        alpha,
        iters,
        X_test,
        y_test,
    )


def load_gpa_data(filename):
    data = pd.read_csv(filename)
    print("Importing from csv, columnns= ", data.columns)
    X_unnormalized = np.array(data["SAT"])
    y = np.array(data["GPA"])

    X_normalized = (X_unnormalized - np.mean(X_unnormalized)) / np.std(X_unnormalized)
    plot_data(X_normalized, y, color="blue", title="SAT vs GPA normalized data")

    X = np.array(X_normalized).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2500
    )
    w = np.zeros(X_train[0].shape[0])
    b = 0
    alpha = 0.001
    iters = 10000
    return (
        X,
        y,
        X_train,
        y_train,
        w,
        b,
        alpha,
        iters,
        X_test,
        y_test,
    )


def load_placement_data(filename):
    data = pd.read_csv(filename)
    print("Importing from csv, columnns= ", data.columns)
    X = np.array(data[["cgpa", "placement_exam_marks"]])
    y = np.array(data["placed"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Performing data transformation ...")
    scaled_dict = {
        "cgpa": X_scaled[:, 0],
        "placement_exam_marks": X_scaled[:, 1],
        "placed": y,
    }
    plot_placement_data(scaled_dict)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=2211
    )
    alpha = 0.01
    iters = 1000
    w = np.zeros(X.shape[1])
    b = 0
    print("Number of 1s:", np.sum(y == 1))
    print("Number of 0s:", np.sum(y == 0))

    return (
        X,
        y,
        X_train,
        y_train,
        w,
        b,
        alpha,
        iters,
        X_test,
        y_test,
    )
