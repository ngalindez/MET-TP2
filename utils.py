import numpy as np
import matplotlib.pyplot as plt
from process_images import *
from numpy_to_pandas import numpy_to_pandas_dataset

def predict(X, w, b):
    z = X @ w + b
    return (np.tanh(z) + 1)/2

def predict_logverosimilitud(X, w, b):
    z = X @ w + b
    return 1 / (1 + np.exp(-z))


def gradiente(X, y, w, b):
    z = X @ w + b
    tanh_z = np.tanh(z)
    f = 0.5 * (tanh_z + 1)
    error = f - y
    sech2 = 1 - tanh_z**2
    
    grad_w = ((error * sech2) @ X)
    grad_b = np.sum(error * sech2)

    return grad_w, grad_b

def gradiente_logverosimilitud(X, y, w, b):
    z = X @ w + b
    f = 1 / (1 + np.exp(-z))
    error = y - f
    grad_w = error @ X
    grad_b = np.sum(error)
    return grad_w, grad_b

def gradient_descent(X_train, y_train, X_test, y_test, alpha, num_it, w=None, b=None):
    # Inicializamos los parámetros
    w = np.random.randn(X_train.shape[1]) * 0.0001 # de una distribución normal
    b = 0.0

    # Para visualizar convergencia
    train_mse_list = []
    train_acc_list = []
    test_mse_list = []
    test_acc_list = []

    for it in range(num_it):
        # Métricas de train
        train_metrics = metricas(X_train, y_train, w, b)
        train_mse_list.append(train_metrics['mse'])
        train_acc_list.append(train_metrics['acc'])
        
        # Métricas de test
        test_metrics = metricas(X_test, y_test, w, b)
        test_mse_list.append(test_metrics['mse'])
        test_acc_list.append(test_metrics['acc'])
        
        grad_w, grad_b = gradiente(X_train, y_train, w, b)

        # Actualizamos parámetros
        w -= alpha * grad_w
        b -= alpha * grad_b
    
        # Vamos imprimiendo el progreso
        if it % 100 == 0 or it == num_it - 1:
            print(f"Iteración {it}: Train MSE={train_metrics['mse']:.4f}, Acc={train_metrics['acc']:.4f} | Test MSE={test_metrics['mse']:.4f}, Acc={test_metrics['acc']:.4f}")

    metrics = {'train_mse_list':train_mse_list, 'train_acc_list':train_acc_list, 'test_mse_list':test_mse_list, 'test_acc_list':test_acc_list}
    return w, b, metrics

def gradient_ascent(X_train, y_train, X_test, y_test, alpha, num_it, w=None, b=None):
    # Inicializamos los parámetros
    w = np.random.randn(X_train.shape[1]) * 0.0001
    b = 0.0

    # Para visualizar convergencia
    train_log_loss_list = []
    test_log_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for it in range(num_it):
        # Métricas de train
        train_metrics = metricas(X_train, y_train, w, b)
        train_log_loss_list.append(train_metrics['log_loss'])
        train_acc_list.append(train_metrics['acc'])
        
        # Métricas de test
        test_metrics = metricas(X_test, y_test, w, b)
        test_acc_list.append(test_metrics['acc'])
        test_log_loss_list.append(test_metrics['log_loss'])
        
        grad_w, grad_b = gradiente_logverosimilitud(X_train, y_train, w, b)

        # Actualizamos parámetros
        w += alpha * grad_w
        b += alpha * grad_b
    
        # Vamos imprimiendo el progreso
        if it % 100 == 0 or it == num_it - 1:
            print(f"Iteración {it}: Train LogLoss={train_metrics['log_loss']:.4f} , Acc={train_metrics['acc']:.4f}| Test LogLoss={test_metrics['log_loss']:.4f}, Acc={test_metrics['acc']:.4f}")

    metrics = {'train_log_loss_list':train_log_loss_list, 'train_acc_list':train_acc_list, 'test_log_loss_list':test_log_loss_list, 'test_acc_list':test_acc_list}
    return w, b, metrics


def plot_metrics(metrics_dict):
    # Ver que métricas estan en el dict
    has_mse = 'train_mse_list' in metrics_dict and 'test_mse_list' in metrics_dict
    has_acc = 'train_acc_list' in metrics_dict and 'test_acc_list' in metrics_dict
    has_log_loss = 'train_log_loss_list' in metrics_dict and 'test_log_loss_list' in metrics_dict
    
    # Calcula la cantidad de subplots necesarios
    num_plots = sum([has_mse, has_acc, has_log_loss])
    
    if num_plots == 0:
        print("No valid metrics found in dictionary")
        return
    
    plt.figure(figsize=(6 * num_plots, 5))
    plot_idx = 1
    
    if has_mse:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(metrics_dict['train_mse_list'], label='Train MSE')
        plt.plot(metrics_dict['test_mse_list'], label='Test MSE')
        plt.xlabel('Iteración')
        plt.ylabel('MSE')
        plt.legend()
        plt.title('Error cuadrático medio')
        plot_idx += 1
    
    if has_log_loss:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(metrics_dict['train_log_loss_list'], label='Train Log Loss')
        plt.plot(metrics_dict['test_log_loss_list'], label='Test Log Loss')
        plt.xlabel('Iteración')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.title('Log Loss')
        plot_idx += 1

    if has_acc:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(metrics_dict['train_acc_list'], label='Train Accuracy')
        plt.plot(metrics_dict['test_acc_list'], label='Test Accuracy')
        plt.xlabel('Iteración')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')
        plot_idx += 1

    plt.tight_layout()
    plt.show()

def metricas(X, y, w, b):
    preds_tanh = predict(X, w, b)
    mse = np.mean((preds_tanh - y) ** 2)
    acc = np.mean((preds_tanh > 0.5) == y)

    preds_sig = predict_logverosimilitud(X, w, b)
    eps = 1e-8  # Para evitar log(0)
    log_loss = -np.mean(y * np.log(preds_sig + eps) + (1 - y) * np.log(1 - preds_sig + eps))
    return {'mse': mse, 'acc': acc, 'log_loss': log_loss}
