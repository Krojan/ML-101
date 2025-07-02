import tensorflow as tf


def normalize_inputs(inputs, X):
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)
    X_normalized = norm_l(inputs)
    return X_normalized
