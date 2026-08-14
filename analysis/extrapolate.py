"""
Log-curve extrapolation of data efficiency results.

Fits Dice = a * ln(size) + b to the efficiency curve data
and projects performance at larger training set sizes.

Usage:
    python analysis/extrapolate.py
"""

import numpy as np
from scipy.optimize import curve_fit

def log_model(x, a, b):
    return a * np.log(x) + b

sizes = np.array([25, 50, 100, 200, 300, 400, 500, 625])

metrics = {
    "Liver Dice": np.array([0.820, 0.846, 0.878, 0.893, 0.898, 0.889, 0.899, 0.899]),
    "Mass Dice": np.array([0.292, 0.358, 0.506, 0.546, 0.559, 0.587, 0.639, 0.652]),
    "Benign Mass": np.array([0.169, 0.147, 0.264, 0.311, 0.317, 0.341, 0.319, 0.403]),
    "Malignant Mass": np.array([0.349, 0.455, 0.618, 0.654, 0.671, 0.701, 0.787, 0.767]),
}

projections = [1000, 2000, 5000]

print("=== Log-curve fit and projections ===\n")

fits = {}
for name, values in metrics.items():
    popt, _ = curve_fit(log_model, sizes, values)
    predicted = log_model(sizes, *popt)
    ss_res = np.sum((values - predicted)**2)
    ss_tot = np.sum((values - np.mean(values))**2)
    r2 = 1 - ss_res / ss_tot
    fits[name] = (popt, r2)
    print(f"{name}: a={popt[0]:.4f}, b={popt[1]:.4f}, R²={r2:.4f}")

print(f"\n{'Size':>6} | {'Liver':>6} | {'Mass':>6} | {'Benign':>7} | {'Malignant':>10}")
print("-" * 50)

for size in list(sizes) + projections:
    label = f"{size}"
    vals = []
    for name in metrics:
        popt = fits[name][0]
        v = min(log_model(size, *popt), 0.99)
        vals.append(v)
    tag = "" if size in sizes else " (proj)"
    print(f"{size:>6} | {vals[0]:>6.3f} | {vals[1]:>6.3f} | {vals[2]:>7.3f} | {vals[3]:>10.3f}{tag}")

print("\n=== Estimated data requirements ===")
for name, target in [("Mass Dice", 0.90), ("Malignant Mass", 0.90), ("Benign Mass", 0.70)]:
    popt = fits[name][0]
    needed = np.exp((target - popt[1]) / popt[0])
    print(f"{name} = {target}: ~{needed:.0f} training images")
