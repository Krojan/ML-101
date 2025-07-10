import numpy as np
import math


def compute_gradient(self):
    dj_dw = np.zeros(self.n)
    dj_db = 0
    for i in range(self.m):
        f_wb = self.compute_y(self.X[i])
        diff = f_wb - self.y[i]
        for j in range(self.n):
            dj_dw[j] += diff * self.X[i, j]
        dj_db += diff

    dj_dw = dj_dw / self.m
    dj_db = dj_db / self.m
    # print(f"Non Regularized : dj_dw = {dj_dw.tolist()}, dj_db = {dj_db}")

    return dj_dw, dj_db


def compute_gradient_with_regularization(self, lamda):
    dj_dw = np.zeros(self.n)
    dj_db = 0
    for i in range(self.m):
        f_wb = self.compute_y(self.X[i])
        diff = f_wb - self.y[i]
        dj_db += diff
        for j in range(self.n):
            w_j = lamda * self.w[j] / self.m
            dj_dw[j] += diff * self.X[i, j] + w_j

    dj_dw = dj_dw / self.m
    dj_db = dj_db / self.m

    # print(f"Regularized : dj_dw = {dj_dw.tolist()}, dj_db = {dj_db}")

    return dj_dw, dj_db


def run(self, regularizaton, lamda, alpha=0.01, num_iterations=1000):
    J_history = []
    P_history = []
    for i in range(num_iterations):
        if regularizaton:
            dj_dw, dj_db = compute_gradient_with_regularization(self, lamda)

        else:
            dj_dw, dj_db = compute_gradient(self)

        for j in range(self.n):
            self.w[j] = self.w[j] - alpha * dj_dw[j]

        self.b = self.b - alpha * dj_db
        cost = (
            self.compute_cost_with_regularization(lamda)
            if lamda > 0
            else self.compute_cost()
        )
        # J_history.append(regularizaton ? self.compute_cost() : self.compute_cost_with_regularization(lamda))
        J_history.append(cost)
        P_history.append([self.w.copy(), self.b])

        if i % math.ceil(num_iterations / 10) == 0:
            print(
                f"Iteration {i:4d}: Cost {J_history[-1]:8.2f} W:{self.w} b:{self.b}   "
            )

    print(f"\nOptimal value of (w,b)= {self.w}, {self.b:8.4f}")

    return self.w, self.b, J_history, P_history
