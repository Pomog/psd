import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

# Define the parameter names (corresponding to the columns)
parameter_names = [
    "remaining MEK / IP.4",
    "Tmin during concentration",
    "Tstart",
    "RPM, crystallization",
    "temp slope"
]

# Input data (X), parameters
X = np.array([
    [3.2739, 59.99, 59.4, 255, -0.4702],
    [2.8076, 55.57, 63.9, 240, -0.4900],
    [2.7847, 62.39, 61.6, 280, -0.4516],
    [2.5559, 55.10, 58.8, 260, -0.43103],
    [2.5063, 50.99, 55.1, 260, -0.3912],
    [2.5080, 51.40, 57.6, 260, -0.4353]
])

# Output data names (corresponding to the columns)
result_names = [
    "Dv 10 v/v %",
    "Dv 50 v/v %",
    "Dv 90 v/v %"
]

# Output data (Y); Dv 10 v/v %, Dv 50 v/v %, Dv 90 v/v %
Y = np.array([
    [4.86, 20.7, 41.4],
    [6.59, 23.0, 45.0],
    [5.07, 20.5, 40.6],
    [4.01, 18.3, 38.6],
    [5.59, 17.9, 34.7],
    [5.88, 18.1, 35.7]
])

if X.shape[0] != Y.shape[0]:
    raise ValueError("Number of samples in X and Y must be the same.")

# Desired output
Y_desired = np.array([[4.0, 15.5, 40]])

if Y_desired.shape[1] != Y.shape[1]:
    raise ValueError("Y_desired dimensions must match Y's output dimensions.")

print("Input data (X) correlation analysis:")
df = pd.DataFrame(X, columns=parameter_names)
correlation_matrix = df.corr()
print("If correlation values (≥0.8) are high between parameters, the model may suffer from multi-collinearity.")
print(correlation_matrix.round(2).to_string())

# Normalize input data (X) using MinMaxScaler
scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

# Principal Component Analysis
pca = PCA()
X_pca = pca.fit_transform(X_normalized)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("\nExplained variance:", pca.explained_variance_)
print("Explained Variance Ratio (cumulative):", cumulative_variance_ratio)

# Normalize output data (Y) using StandardScaler
scaler_Y = StandardScaler()
Y_normalized = scaler_Y.fit_transform(Y)

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_normalized)

# Fit Linear Regression Model
model = LinearRegression()
model.fit(X_poly, Y_normalized)


# Objective Function
def objective(X_input):
    # Polynomial transformation of input
    X_poly_input = poly.transform(X_input.reshape(1, -1))
    Y_pred_normalized = model.predict(X_poly_input)
    Y_pred = scaler_Y.inverse_transform(Y_pred_normalized)
    error = np.mean((Y_pred - Y_desired) ** 2)
    return error


# We'll use min and max normalized parameters as boundaries
X_min = np.min(X_normalized, axis=0)
X_max = np.max(X_normalized, axis=0)

# Initial Guess and Bounds
initial_guess = np.mean(X_normalized, axis=0)
bounds = [
    (X_min[0], X_max[0]),  # 'remaining MEK / IP.4'
    (X_min[1], X_max[1]),  # 'Tmin during concentration'
    (None, None),  # 'Tstart'
    (None, None),  # 'RPM, crystallization'
    (None, None),  # 'temp slope'
]

# Perform Optimization
result = minimize(
    objective,
    initial_guess,
    method='L-BFGS-B',
    bounds=bounds,
    options={'ftol': 1e-5, 'maxiter': 1000}
)

# Check Optimization Result
if result.success:
    X_optimized_normalized = result.x
    X_optimized = scaler_X.inverse_transform(X_optimized_normalized.reshape(1, -1))
    print("\nOptimized Input Parameters:")
    for param_name, value in zip(parameter_names, X_optimized[0]):
        print(f"{param_name}: {value:.4f}")
else:
    print("Optimization failed:", result.message)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Input Parameters")
plt.show()

# PCA Explained Variance Plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, alpha=0.7, label="Individual")
plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio * 100, where="mid", color='red',
         label="Cumulative")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.title("PCA Explained Variance")
plt.legend()
plt.show()

# Boxplot of Input Data (Normalized)
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(X_normalized, columns=parameter_names))
plt.title("Normalized Input Data Distribution")
plt.ylabel("Normalized Value")
plt.xticks(rotation=45)
plt.show()

# Boxplot of Output Data (Normalized)
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(Y_normalized, columns=result_names))
plt.title("Normalized Output Data Distribution")
plt.ylabel("Normalized Value")
plt.xticks(rotation=45)
plt.show()

# Scatter Plot of Actual vs Predicted Outputs
Y_pred_normalized = model.predict(X_poly)
Y_pred = scaler_Y.inverse_transform(Y_pred_normalized)

for i, name in enumerate(result_names):
    plt.figure(figsize=(8, 6))
    plt.scatter(Y[:, i], Y_pred[:, i], alpha=0.7, label="Predicted vs Actual")
    plt.plot([min(Y[:, i]), max(Y[:, i])], [min(Y[:, i]), max(Y[:, i])], color='red', linestyle='--', label="Ideal")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted for {name}")
    plt.legend()
    plt.show()

# Optimization Error Convergence
if result.success:
    optimization_error = [objective(result.x)]
    plt.figure(figsize=(8, 6))
    plt.plot(optimization_error, marker='o', label="Optimization Error")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Optimization Convergence")
    plt.legend()
    plt.show()
else:
    print("Optimization error convergence plot skipped as optimization failed.")
