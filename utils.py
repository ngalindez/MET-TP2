import numpy as np
import matplotlib.pyplot as plt
from process_images import *
from numpy_to_pandas import numpy_to_pandas_dataset

def predict(X, w, b):
    z = X @ w + b
    return (np.tanh(z) + 1)/2


def gradiente(X, y, w, b):
    z = X @ w + b
    tanh_z = np.tanh(z)
    f = 0.5 * (tanh_z + 1)
    error = f - y
    sech2 = 1 - tanh_z**2
    
    grad_w = ((error * sech2) @ X)
    grad_b = np.sum(error * sech2)

    return grad_w, grad_b

def gradient_descent(X_train, y_train, X_test, y_test, alpha, num_it, w=None, b=None):
    # Inicializamos los parámetros
    w = np.random.randn(X_train.shape[1]) * 0.0001
    b = 0.0

    # Para visualizar convergencia
    train_mse_list = []
    train_acc_list = []
    test_mse_list = []
    test_acc_list = []

    for it in range(num_it):
        # Métricas de train
        train_mse, train_acc = metricas(X_train, y_train, w, b)
        train_mse_list.append(train_mse)
        train_acc_list.append(train_acc)
        
        # Métricas de test
        test_mse, test_acc = metricas(X_test, y_test, w, b)
        test_mse_list.append(test_mse)
        test_acc_list.append(test_acc)
        
        grad_w, grad_b = gradiente(X_train, y_train, w, b)

        # Actualizamos parámetros
        w -= alpha * grad_w
        b -= alpha * grad_b
    
        # Vamos imprimiendo el progreso
        if it % 100 == 0 or it == num_it - 1:
            print(f"Iteración {it}: Train MSE={train_mse:.4f}, Acc={train_acc:.4f} | Test MSE={test_mse:.4f}, Acc={test_acc:.4f}")

    return w, b, train_mse_list, train_acc_list, test_mse_list, test_acc_list


def plot_metrics(train_mse, train_acc, test_mse, test_acc):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_mse, label='Train MSE')
    plt.plot(test_mse, label='Test MSE')
    plt.xlabel('Iteración')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Error cuadrático medio')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.xlabel('Iteración')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.show()

def metricas(X, y, w, b):
    preds = predict(X, w, b)
    mse = np.mean((preds - y) ** 2)
    acc = np.mean((preds > 0.5) == y)
    return mse, acc