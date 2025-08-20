# L1-Norm Support Vector Machine (SVM) Classifier

**Repository:** L1-norm SVM implemented with `cvxpy` for binary classification (sparse-encouraging regularization).  
**Goal:** Train an SVM that minimizes the L1-norm of the weight vector, check linear separability of a dataset, and identify support vectors.

---

## Table of contents
- [Overview](#overview)  
- [Problem statement](#problem-statement)  
- [Repository structure](#repository-structure)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Outputs (generated files)](#outputs-generated-files)  
- [Algorithm & implementation details](#algorithm--implementation-details)  
- [Support vector & separability rules](#support-vector--separability-rules)  
- [Troubleshooting & tips](#troubleshooting--tips)  
- [Reproducibility](#reproducibility)  
- [License / Contact](#license--contact)

---

# Overview

This project implements a soft-margin binary SVM that uses the **L1-norm** (`||w||_1`) on the weight vector `w` instead of the usual L2 regularizer. L1 regularization often encourages sparsity in `w`. The optimization is solved with `cvxpy`. Two datasets are provided:

- `train_ls.csv` — linearly separable dataset  
- `train_nls.csv` — non-linearly separable dataset

You **must not** change the provided data files.

---

# Problem statement

We solve the following optimization:

minimize over `w, b, ξ`  
\[
\frac{1}{2}\|w\|_1 + C \sum_{i=1}^n \xi_i
\]

subject to  
\[
y_i (w\cdot x_i + b) \ge 1 - \xi_i,\quad \xi_i \ge 0 \quad \forall i
\]

- `C` is the regularization parameter (fixed in this project as `C = 1`).
- `ξ` are slack variables for soft margin.
- Labels are converted from `{0,1}` to `{-1,1}` for SVM constraints.

---

# Repository structure

```
.
├── svm.py                  # Main training script
├── train_ls.csv            # Linearly separable dataset (DO NOT MODIFY)
├── train_nls.csv           # Non-linearly separable dataset (DO NOT MODIFY)
├── weights_ls.json         # Provided reference weights/bias for 'ls' (do not change)
├── weights_nls.json        # Provided reference weights/bias for 'nls' (do not change)
├── sv_ls.json              # Provided separability info/support vectors for 'ls' (do not change)
└── sv_nls.json             # Provided separability info/support vectors for 'nls' (do not change)
```

Running `svm.py` will create two new files per dataset (see **Outputs** below).

---

# Prerequisites

Python 3.8+ recommended.

Install required packages:

```bash
pip install numpy pandas scikit-learn cvxpy
```

`cvxpy` requires a solver — common choices include `ECOS`, `OSQP`, or `SCS`. If you encounter solver errors, install one:

```bash
pip install ecos osqp scs
```

---

# Installation

Clone your repository and make sure `svm.py` and the CSV data files are present:

```bash
git clone <your-repository-url>
cd <repository-directory>
```

---

# Usage

Run training on the linearly separable dataset:

```bash
python svm.py train_ls.csv
# Generates: weight_svm2_ls.json and sv_svm2_ls.json
```

Run training on the non-linearly separable dataset:

```bash
python svm.py train_nls.csv
# Generates: weight_svm2_nls.json and sv_svm2_nls.json
```

The script accepts **one** command-line argument: the path to the CSV file.

---

# Outputs (generated files)

For an input file named `train_<prefix>.csv` the script writes:

1. `weight_svm2_<prefix>.json` — learned model parameters:

```json
{
  "weights": [w_0, w_1, ..., w_{d-1}],
  "bias": b
}
```

2. `sv_svm2_<prefix>.json` — separability flag and support vector indices (0-based indices referring to the original CSV ordering):

```json
{
  "separable": 1,
  "support_vectors": [i, j, k]
}
```

- `separable` is `1` if dataset is found linearly separable (per the tolerance rule below), otherwise `0`.
- For non-separable datasets, `support_vectors` will be an empty list.

---

# Algorithm & implementation details

1. **Data loading**  
   - `load_data_csv(filename)` reads the CSV with `pandas`.
   - Features `X` and labels `y` are separated.
   - Labels `{0,1}` are mapped to `{-1,+1}` for SVM constraints.

2. **Preprocessing**  
   - Features are standardized using `sklearn.preprocessing.StandardScaler` (zero mean, unit variance). Standardization is important so the solver behaves well and margins are meaningful.

3. **Optimization** (`train_svm_func`)  
   - Optimization variables: `w` (shape `d`), `b` (scalar), `xi` (shape `n`) implemented in `cvxpy`.
   - Objective: minimize `0.5 * norm1(w) + C * sum(xi)` with `C = 1`.
   - Constraints:
     - `y_i * (w @ x_i + b) >= 1 - xi_i` for every `i`.
     - `xi_i >= 0` for every `i`.
   - Solve with a `cvxpy` solver (ECOS/OSQP/SCS). If solver fails, try different solver options.

4. **Separable check**  
   - After solving, the script checks slack variables `xi`. If all `xi_i <= tol` (with `tol = 1e-4`), dataset is marked **linearly separable**.

5. **Support vector identification**  
   - For separable data: a data point `i` is considered a support vector if
     \[
     \big| y_i (w\cdot x_i + b) - 1 \big| \le 10^{-4}
     \]
   - For non-separable data: `support_vectors` is set to `[]`.

---

# Support vector & separability rules

- `C = 1` (fixed).
- Tolerance for slack to determine separability: `tol = 1e-4`.
- Tolerance to identify support vectors on the margin: `1e-4`.

These tolerances are chosen to account for numerical solver precision.

---

# Troubleshooting & tips

- If `cvxpy` raises an error about missing solvers, install `ecos`, `osqp`, or `scs` via pip.
- If the solver status is not `optimal` (e.g., `infeasible`, `unbounded`, or `user_limit`), check:
  - Data format (no missing values; correct label mapping).
  - Feature scaling; confirm `StandardScaler` has been applied.
  - Try a different solver: `cvxpy.Problem(...).solve(solver=cvxpy.ECOS)` or `OSQP`.
- Numerical tolerances (`1e-4`) are conservative; if you observe borderline cases you can inspect `xi` values and decision scores directly.

---

# Reproducibility

- The optimization is deterministic given the same solver and version; however, different solvers or solver versions may yield slightly different numeric results. For reproducible outcomes, prefer the same solver and package versions.
- Do not modify the CSV data files (`train_ls.csv`, `train_nls.csv`) — the evaluation and provided reference JSON files depend on them.

---

# Notes

- This implementation intentionally uses L1 regularization (`||w||_1`) instead of the classical L2 (`||w||_2^2`). L1 tends to encourage sparse weight vectors which can be useful for feature selection and interpretability.
- The code saves human-readable JSON results for easy grading and inspection.

---

# License & contact

This project contains example code for educational purposes. Modify and reuse as permitted by your institutional or project guidelines.

If you need help running the script or want a walk-through of the solver output, share the solver error/status and I can suggest the next steps.
