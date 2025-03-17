import numpy as np
import pywt
import matplotlib.pyplot as plt
from pyswarm import pso
from sklearn.metrics import mean_squared_error
from sklearn.cluster import OPTICS


import pandas as pd
data = pd.read_csv("EMRtest.csv")
import numpy as np

time_series_data = np.array(data["data"])
time = np.linspace(0, 1, len(time_series_data))

def wavelet_denoising(data, wavelet, level, threshold):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    thresholded_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(thresholded_coeffs, wavelet)
def apply_optics(data, eps, min_samples):
    optics = OPTICS(min_samples=10, xi=0.001, min_cluster_size=12, metric='euclidean')
    labels = optics.fit_predict(data)
    return labels

def compute_noise_and_clusters(labels):
    noise_points = np.sum(labels == -1)
    cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
    return noise_points / len(labels), cluster_count
def objective_function(data):
    eps, min_samples = np.inf,17
    if eps <= 0 or min_samples < 2:
        return np.inf
    labels = apply_optics(data, eps, min_samples)
    noise_ratio, cluster_count = compute_noise_and_clusters(labels)
    return (noise_ratio + (1 / (cluster_count + 1e-6))),noise_ratio 
threshold_values = []
level_values = []
iteration_numbers = []
loss_value = []
loss_rotao = []

def optimize_wavelet_params(data):
    lower_bounds = [0, 0]  
    upper_bounds = [150, 10] 
    def wrapped_objective_function(params):
        
        denoised_signal = wavelet_denoising(data, 'db4', int(params[1]), params[0])
        w = 3 
        n = len(denoised_signal)
        data_denosied = np.array([denoised_signal[i:i+w] for i in range(n-w+1)])
        value,noise_ratio = objective_function(data_denosied)
        threshold_values.append(params[0])
        level_values.append(params[1])
        iteration_numbers.append(len(iteration_numbers) + 1)
        loss_value.append(value)
        loss_rotao.append(noise_ratio)
        iter = len(iteration_numbers) + 1
        print(f'iteration_numbers：{iter} Parameters - eps: {params[0]}, min_samples: {params[1]},allparam{params}')
        return value

    optimized_params, fopt = pso(wrapped_objective_function, lower_bounds, upper_bounds, 
                                 swarmsize=30, maxiter=150, minstep=1e-11, minfunc=1e-11)
    return optimized_params

def plot_time_series(original_data, cleaned_data):
    time_indices = np.arange(original_data.shape[0])
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(time_indices, original_data, marker='o', linestyle='-', color='b')
    plt.title('Original Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    plt.subplot(1, 2, 2)
    plt.plot(time_indices, cleaned_data, marker='o', linestyle='-', color='g')
    plt.title('Denoised Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plt.show()

def plot_optimization_process():
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(iteration_numbers, threshold_values, marker='o', linestyle='-', color='b')
    plt.title('Threshold Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Threshold')

    plt.subplot(1, 2, 2)
    plt.plot(iteration_numbers, level_values, marker='o', linestyle='-', color='g')
    plt.title('Level Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Level')

    plt.tight_layout()
    plt.show()

best_params = optimize_wavelet_params(time_series_data)
best_threshold, best_level = best_params
print(f'Optimized Parameters - Threshold: {best_threshold}, Level: {best_level}')

denoised_data = wavelet_denoising(time_series_data, 'db4', int(best_level), best_threshold)

iter_out = pd.DataFrame({"iteration_numbers":iteration_numbers,"threshold_values":threshold_values,"level_values":level_values,"loss_value":loss_value,"loss_rotao":loss_rotao})
iter_out.to_excel("optim_result.xlsx", index=False)

time_indices = np.arange(time_series_data.shape[0])
denoise_out = pd.DataFrame({"time_series_data":time_series_data,"denoised_data":denoised_data})
denoise_out.to_excel("denoise_out_result.xlsx", index=False)

plot_time_series(time_series_data, denoised_data)

plot_optimization_process()
