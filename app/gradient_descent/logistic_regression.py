import numpy as np
import gradient_descent as gradient_descent

np.set_printoptions(precision=5, suppress=True)


class logistic_regression:
    def __init__(self, X, y, w_init, b_init):
        self.m = X.shape[0]  # number of training examples
        self.n = X.shape[1]  # number of features
        self.X = X
        self.y = y
        self.w = w_init
        self.b = b_init

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # protect against overflow
        g = 1.0 / (1.0 + np.exp(-z))
        return g

    def compute_y(self, x):
        z = np.dot(self.w, x) + self.b
        return self.sigmoid(z)

    def compute_cost(self):
        cost = 0
        for i in range(self.m):
            f_wb = self.compute_y(self.X[i])
            cost_i = -self.y[i] * np.log(f_wb) - (1 - self.y[i]) * np.log(1 - f_wb)
            cost += cost_i

        cost = cost / self.m
        return cost

    def compute_cost_with_regularization(self, lamda=0.7):
        mean_sq_cost = self.compute_cost()
        regularization_cost = 0

        for j in range(self.n):
            regularization_cost += self.w[j] ** 2
        regularization_cost = regularization_cost * lamda / (2 * self.m)

        return mean_sq_cost + regularization_cost

    def train(self, alpha, num_iterations, regularization=False, lamda=0.07):
        w, b, J_history, P_history = gradient_descent.run(
            self,
            alpha=alpha,
            num_iterations=num_iterations,
            regularizaton=regularization,
            lamda=lamda,
        )
        return w, b, J_history, P_history

    def predict(self, X, Y_target, threshold=0.5):
        m = X.shape[0]
        y_predictions = []
        for i in range(m):
            y = self.compute_y(X[i])
            if y > threshold:
                y = 1
            else:
                y = 0

            print(f"prediction: {y:0.2f}, target value: {Y_target[i]}")
            y_predictions.append(y)

        return y_predictions
