import load_coffee_data as dataset_generator
import normalizer
import numpy as np


def get_model_params():
    W1 = np.array([[-8.93, 0.29, 12.9], [-0.1, -7.32, 10.81]])
    b1 = np.array([-9.82, -9.28, 0.96])
    W2 = np.array([[-31.18], [-27.59], [-32.56]])
    b2 = np.array([[15.41]])
    W = [W1, W2]
    b = [b1, b2]
    return W, b


def get_normalized_examples():
    X, Y = dataset_generator.load_coffee_data()
    inputs = np.array([[200, 13.9], [200, 17]])  # postive example  # negative example
    normalized_inputs = normalizer.normalize_inputs(inputs, X)
    return normalized_inputs
