import numpy as np
from sklearn.preprocessing import StandardScaler

# Experiment parameters: solvent/reagent mass, temperature C, catalyst amount %
X = np.array([
    [10, 20, 1],
    [10, 30, 1],
    [10, 20, 10],
    [10, 50, 5],
    [5, 50, 5],
    [5, 50, 2],
    [7, 40, 10]
])

# Output results: yield %, purity %
Y = np.array([
    [80, 80],
    [85, 75],
    [75, 90],
    [70, 50],
    [60, 45],
    [95, 40],
    [90, 78]
])


print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
print("Experiment parameters. Raw data\n", X)
print("Experiment results. Raw data\n", X)

# Scaling the input date to have similar ranges, using StandardScaler
# Mean = 0
# Standard deviation = 1
# putting all features on a comparable scale

# Create scalers
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Normalize X and Y
X_normalized = scaler_X.fit_transform(X)
Y_normalized = scaler_Y.fit_transform(Y)

print("Normalized Experiment parameters:\n", X_normalized)
print("Normalized Experiment results:\n", Y_normalized)

print("Mean of X (normalized):", np.mean(X_normalized, axis=0))
print("Std of X (normalized):", np.std(X_normalized, axis=0))
print("Mean of Y (normalized):", np.mean(Y_normalized, axis=0))
print("Std of Y (normalized):", np.std(Y_normalized, axis=0))
