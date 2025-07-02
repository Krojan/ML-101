---

# ML-101

This project implements the machine learning foundations learned from the Coursera Machine Learning Specialization Course by DeepLearning.AI & Stanford University. It contains two sections - **Regression model** and **Tensorflow model**.

### Regression model

Implements gradient descent algorithm for cost minimization. Four options available:

#### Usage

```bash
python3 app/gradient_descent/index.py logistic         # Runs logistic regression
python3 app/gradient_descent/index.py linear-single    # Runs linear regression on single feature input
python3 app/gradient_descent/index.py linear-multiple  # Runs linear regression on multi-feature input
python3 app/gradient_descent/index.py salary           # Runs linear regression on salary dataset
python3 app/gradient_descent/index.py regular          # Runs both linear and logistic regression with regularization
```

### Tensorflow model

Mimicks tensorflow model in minimal setting - how layers are connected and the outputs are feed onto one another.

#### Usage

_Preconditions_ : Value of W and b.

```python
model = tf_model(W, b)
model.summary()
model.predict()
```

## Installation

```bash
pip install -r requirements.txt
```
