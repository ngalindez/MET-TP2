import numpy as np
import matplotlib.pyplot as plt
from process_images import *
from numpy_to_pandas import numpy_to_pandas_dataset

def predict(X, w, b):
    z = X @ w + b
    return (np.tanh(z) + 1)/2

def gradiente(X, y, w, b):
    f = predict(X, w, b)
    error = f - y
    z = X @ w + b
    sech2 = 1 - np.tanh(z)**2

    grad_w = (error * (sech2)) @ X
    grad_b = np.sum(error * (sech2))

    return grad_w, grad_b

def loss(y_pred, y_true):
    return np.sum((y_pred - y_true) ** 2)

def gradient_descent(X, y, alpha, num_it, w=None, b=None):
    # Inicializamos los parámetros
    w = np.random.randn(X.shape[1]) * 0.01
    b = 0.0

    # Para visualizar convergencia
    loss_history = []

    for it in range(num_it):
        y_pred = predict(X, w, b)
        loss_value = loss(y_pred, y)
        loss_history.append(loss_value)
        grad_w, grad_b = gradiente(X, y, w, b)

        # Actualizamos parámetros
        w -= alpha * grad_w
        b -= alpha * grad_b
    
        # Vamos imprimiendo el progreso
        if it % 100 == 0:
            print(f"Iteración {it}: Loss = {loss_value:.4f}")

    return w, b, loss_history


def plot_loss_curve(loss_history, log_scale=False):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Pérdida (loss)')
    plt.xlabel('Iteración')
    plt.ylabel('Error cuadrático medio')
    plt.title('Curva de pérdida durante el entrenamiento')
    plt.legend()
    if log_scale: plt.yscale('log')
    plt.grid()
    plt.show()

def compute_metrics(X, y, w, b):
    """
    Calcula error cuadrático medio y accuracy.
    Args:
        X: array (N, D)
        y: array (N,)
        w: vector pesos
        b: bias
    Returns:
        mse: error cuadrático medio
        acc: accuracy
    """
    preds = predict(X, w, b)
    mse = np.mean((preds - y) ** 2)
    acc = np.mean((preds > 0.5) == y)
    return mse, acc