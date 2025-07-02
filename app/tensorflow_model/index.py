import preprocess as data_preprocessor
from tensorflow_model import tf_model

W, b = data_preprocessor.get_model_params()
inputs = data_preprocessor.get_normalized_examples()

model = tf_model(W, b)
model.summary()
predictions = model.predict(inputs)
print("\nPredictions=", predictions)
