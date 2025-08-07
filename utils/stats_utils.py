# utils/stats_utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats

def is_normal_distribution(data, alpha=0.05):
    stat, p = stats.shapiro(data.dropna())
    return p > alpha

def apply_clt(data, sample_size=30, n_simulations=1000):
    means = []
    for _ in range(n_simulations):
        sample = data.dropna().sample(n=sample_size, replace=True)
        means.append(sample.mean())
    return means

def plot_clt(data, col, output_dir):
    clt_means = apply_clt(data[col])
    plt.figure(figsize=(6, 4))
    sns.histplot(clt_means, kde=True)
    plt.title(f"CLT Simulation for {col}")
    path = os.path.join(output_dir, f"clt_{col}.png")
    plt.savefig(path)
    plt.close()
    return path

def simulate_confidence_intervals(data, col, confidence=0.95, sample_size=30, n_simulations=100):
    means, lower_bounds, upper_bounds = [], [], []
    for _ in range(n_simulations):
        sample = data[col].dropna().sample(n=sample_size, replace=True)
        sample_mean = sample.mean()
        sample_std = sample.std()
        se = sample_std / np.sqrt(sample_size)
        t_crit = stats.t.ppf((1 + confidence) / 2, df=sample_size - 1)
        lower = sample_mean - t_crit * se
        upper = sample_mean + t_crit * se
        means.append(sample_mean)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    return means, lower_bounds, upper_bounds

def plot_confidence_intervals(means, lower_bounds, upper_bounds, population_mean, col, output_dir):
    plt.figure(figsize=(8, 6))
    for i in range(len(means)):
        color = 'red' if lower_bounds[i] > population_mean or upper_bounds[i] < population_mean else 'black'
        plt.plot([lower_bounds[i], upper_bounds[i]], [i, i], color=color)
        plt.plot(means[i], i, 'bo')
    plt.axvline(population_mean, color='green', linestyle='--', label='Population Mean')
    plt.xlabel('Confidence Interval Range')
    plt.ylabel('Simulation')
    plt.title(f"Confidence Intervals for {col}")
    plt.legend()
    path = os.path.join(output_dir, f"ci_{col}.png")
    plt.savefig(path)
    plt.close()
    return path
