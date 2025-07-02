import numpy as np
import gradient_descent as gradient_descent

np.set_printoptions(precision=5, suppress=True)


class linear_regression:
    def __init__(self, X, y, w_init, b_init):
        self.m = X.shape[0]  # number of training examples
        self.n = X.shape[1]  # number of features
        self.X = X
        self.y = y
        self.w = w_init
        self.b = b_init

    def compute_y(self, x):
        return np.dot(self.w, x) + self.b

    def compute_cost(self):
        cost = 0
        for i in range(self.m):
            f_wb = self.compute_y(self.X[i])
            diff = f_wb - self.y[i]
            err = diff**2
            cost += err

        cost = cost / (2 * self.m)
        return cost

    def compute_cost_with_regularization(self, lamda=0.7):
        mean_sq_cost = self.compute_cost()
        regularization_cost = 0

        for j in range(self.n):
            regularization_cost += self.w[j] ** 2
        regularization_cost = regularization_cost * lamda / (2 * self.m)
        return mean_sq_cost + regularization_cost

    def train(self, alpha, num_iterations, regularization=False, lamda=0.7):
        w, b, J_history, P_history = gradient_descent.run(
            self,
            alpha=alpha,
            num_iterations=num_iterations,
            regularizaton=regularization,
            lamda=lamda,
        )
        return self.w, self.b, J_history, P_history

    def predict(self, X, Y_target):
        m = X.shape[0]
        for i in range(m):
            y = self.compute_y(X[i])
            print(f"prediction: {y:0.2f}, target value: {Y_target[i]}")
