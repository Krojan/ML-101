import numpy as np
from prettytable import PrettyTable


class tf_model:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        g = 1 / (1 + np.exp(-z))
        return g

    def dense(self, a_in, layer):
        # layer = index of layer used to ref W
        # get W and b for this layer
        w_layer = self.W[layer]
        b_layer = self.b[layer]
        output_dim = w_layer.shape[1]
        layer_out = np.zeros(output_dim)
        for i in range(output_dim):
            print("\t \t Unit: ", i + 1)
            z = np.dot(a_in, w_layer[:, i]) + b_layer[i]
            print(f"\t \t Non Sigmoid Output = {z} \n")
            layer_out[i] = self.sigmoid(z)

        print(f"\t \t Output: ", layer_out)
        return layer_out

    def sequential(self, x_in):
        layer_in = x_in
        num_layers = len(self.W)
        for layer in range(num_layers):
            print(f"\t Layer: {layer + 1}")
            print("\t Inputs = ", layer_in)
            layer_out = self.dense(layer_in, layer)
            layer_in = layer_out
        return layer_out

    def predict(self, examples):
        num_examples = len(examples)
        predictions = np.zeros(num_examples)
        # for each input, run calculations
        for i in range(num_examples):
            print("\nRunning prediction for ", examples[i])
            predictions[i] = self.sequential(examples[i])
        return predictions

    def summary(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Type", "Output Shape", "Params"]
        num_layers = len(self.W)
        for layer_idx in range(num_layers):
            layer_name = f"layer_{layer_idx+1}"
            type = "Dense"
            w_layer = self.W[layer_idx]
            input_dim = w_layer.shape[0]
            output_dim = w_layer.shape[1]
            output_shape = f"None(, {output_dim})"
            params = input_dim * output_dim + output_dim
            table.add_row([layer_name, type, output_shape, params])

        print(table)
