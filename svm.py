#Final
import sys
import pandas as pd
import numpy as np
import cvxpy as cp
import json
import os
from sklearn.preprocessing import StandardScaler

# LOAD FILE FROM CSV FILE
def load_data_csv(filename):
    data = pd.read_csv(filename)
    x_data = data.iloc[:, :-1].values  # Features
    y_data = data.iloc[:, -1].values  # Labels
    y_data = np.where(y_data == 0, -1, 1)  # Convert labels to -1 and 1 for SVM

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    return x_data, y_data

# TRAIN SVM FUNCTION
# INPUT: X,Y,C
def train_svm_func(x, y, C=1, tol=1e-4):
    n, d = x.shape
    w = cp.Variable(d)
    b = cp.Variable()
    xi = cp.Variable(n)

    # objective function
    objective_func = cp.Minimize(0.5 * cp.norm1(w) + C * cp.sum(xi))

    # constraints
    constraints_func = [y[i] * (x[i] @ w + b) >= 1 - xi[i] for i in range(n)]
    constraints_func += [xi >= 0]

    # Problem setup
    problem = cp.Problem(objective_func, constraints_func)
    problem.solve()

    # weights, bias, and slack variables
    weights = w.value
    bias = b.value
    xi_vals = xi.value

    # Check if linearly separable
    separable = int(np.all(xi_vals <= tol))

    # Determine support vectors only if the dataset is linearly separable
    support_vectors = (
        [i for i in range(n) if abs(y[i] * (x[i] @ weights + bias) - 1) <= tol] if separable else []
    )

    return weights, bias, separable, support_vectors

# SAVE RESULTS FUNCTION
def save_results_func(weights, bias, separable, support_vectors, prefix):
    # weights are converted to a list if it's a numpy array
    weight_data = {
        "weights": weights.tolist() if isinstance(weights, np.ndarray) else weights,
        "bias": float(bias)  # Convert bias to float if it's a numpy.float64 or other non-JSON serializable type
    }
    with open(f"weight_svm2_{prefix}.json", "w") as f:
        json.dump(weight_data, f, indent=2)  # Add indent for pretty printing

    # Save separable status and support vectors
    svm_data = {
        "separable": separable,
        "support_vectors": support_vectors
    }
    with open(f"sv_svm2_{prefix}.json", "w") as f:
        json.dump(svm_data, f, indent=2)  # Add indent for pretty printing

# MAIN FUNCTION
def main():
    if len(sys.argv) != 2:
        print("Usage: python svm.py train_<dataset_name>.csv")
        sys.exit(1)

    filename = sys.argv[1]
    prefix = os.path.splitext(os.path.basename(filename))[0].replace("train_", "")

    # Load data
    x, y = load_data_csv(filename)

    # Train SVM
    weights, bias, separable, support_vectors = train_svm_func(x, y)

    # Save results
    save_results_func(weights, bias, separable, support_vectors, prefix)

if __name__ == "__main__":
    main()