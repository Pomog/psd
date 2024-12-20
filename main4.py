import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Define the parameter names (corresponding to the columns)
parameter_names = [
    "remaining MEK / IP.4",
    "Tmin during concentration",
    "Tstart",
    "RPM, crystallization",
    "temp slope",
    "start cooling rate slope",
    "middle cooling rate slope",
    "final cooling rate slope"
]

# input data (X), parameters
X = np.array([
    [3.2739, 59.99, 59.4, 255, -0.4702, 0.0168, 0.0008, 0.0032],
    [2.8076, 55.57, 63.9, 240, -0.4900, 0.0201, 0.0012, -0.0002],
    [2.7847, 62.39, 61.6, 280, -0.4516, -0.0191, 0.0079, 0.0006],
    [2.5559, 55.10, 58.8, 260, -0.4310, 0.0103, 0.0017, -0.0043],
    [2.5063, 50.99, 55.1, 260, -0.3912, 0.0149, 0.0031, -0.0071],
    [2.5080, 51.40, 57.6, 260, -0.4353, 0.0138, 0.0001, -0.0048]
])

# output data names (corresponding to the columns)
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

print("input data (X) correlation analysis")
df = pd.DataFrame(X, columns=parameter_names)
correlation_matrix = df.corr()
print("if correlation values (≥0.8) are high between parameters, the model may suffer from multi-collinearity.")
print(correlation_matrix.round(2).to_string())

# Normalize input data (X) using StandardScaler
scaler_X = RobustScaler()
X_normalized = scaler_X.fit_transform(X)

# Principal Component Analysis
pca = PCA(n_components=None)
# Sensitive to outliers, they can distort the variance.
X_pca = pca.fit_transform(X_normalized)

print("Explained variance:", pca.explained_variance_)
print("X_PCA:", X_pca)

num_components = sum(pca.explained_variance_ > 1)
print(f"Number of components selected by Kaiser Criterion: {num_components}")

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("Explained Variance Ratio (individual components):", explained_variance_ratio)
print("Cumulative Variance Ratio:", cumulative_variance_ratio)

variance_first_three = cumulative_variance_ratio[2] * 100
print(f"The first three components explain approximately {variance_first_three:.2f}% of the variance.")

# We'll use min and max normalized parameters as boundaries
X_min = np.min(X_normalized, axis=0)
X_max = np.max(X_normalized, axis=0)

# Normalize output data (Y) using StandardScaler
scaler_Y = StandardScaler()
Y_normalized = scaler_Y.fit_transform(Y)

# The LinearRegression is used because it is easy to get the weight coefficients
model = LinearRegression()
model.fit(X_normalized, Y_normalized)

print("\nWeight coefficients (coefficients for the normalized data of the linear regression model):")
for param_name, coef in zip(parameter_names, model.coef_[0]):
    print(f"{param_name}: {coef:.4f}")

# Accuracy of the model
print("\n# Accuracy of the model")
Y_pred_normalized = model.predict(X_normalized)
Y_pred = scaler_Y.inverse_transform(Y_pred_normalized)
print("Predicted results:")
print(Y_pred)
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)
print(f"MSE: {mse}, R^2: {r2}", "\n")

residuals = Y - Y_pred
plt.scatter(Y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Linear model Residuals plot")
plt.show()

for i, param_name in enumerate(parameter_names):
    plt.figure(figsize=(10, 6))
    for j, result_name in enumerate(result_names):
        sorted_indices = np.argsort(X[:, i])  # Sort data by the X-axis parameter
        X_sorted = X[sorted_indices, i]
        Y_sorted = Y[sorted_indices, j]
        Y_pred_sorted = Y_pred[sorted_indices, j]

        plt.scatter(X_sorted, Y_sorted, label=f"{param_name} vs {result_name}")
        plt.plot(X_sorted, Y_pred_sorted, color='red', label=f"Prediction for {result_name}")

    plt.xlabel(param_name)
    plt.ylabel("Y values")
    plt.title(f"Dependency of {param_name} on Y")
    plt.legend()
    plt.show()


# The function returns the error between the predicted output and Y_desired
def objective(X_input):
    # X_input will be in the normalized space, so we need to predict in normalized space first
    X_input_reshaped = X_input.reshape(1, -1)  # Reshape to match the input shape
    Y_pred_normalized_o = model.predict(X_input_reshaped)

    # Reverse the normalization to get the predicted output in the original scale
    Y_pred_o = scaler_Y.inverse_transform(Y_pred_normalized_o)

    # Calculate the error
    error = np.mean((Y_pred_o - Y_desired) ** 2)
    print("error: ", error)
    return error


initial_guess = np.mean(X_normalized, axis=0)

# None means that parameter can be outside the input data range, predicted based on the regression model
bounds = [
    (X_min[0], X_max[0]),  # 'remaining MEK / IP.4'
    (None, None),  # 'Tmin during concentration'
    (X_min[2], X_max[2]),  # 'Tstart'
    (X_min[3], X_max[3]),  # 'RPM, crystallization'
    (None, None),  # 'temp slope'
    (None, None),  # 'start cooling rate slope'
    (None, None),  # 'middle cooling rate slope'
    (None, None)  # 'final cooling rate slope'
]

# Perform the optimization to find the best input parameters
result = minimize(
    objective,
    initial_guess,
    method='L-BFGS-B',
    bounds=bounds,
    options={'ftol': 1e-5, 'maxiter': 1000}
)

# Check if the optimization was successful
if result.success:
    # Get the optimized input parameters (normalized)
    X_optimized_normalized = result.x

    # Reverse the normalization to get the optimized input in the original scale
    X_optimized = scaler_X.inverse_transform(X_optimized_normalized.reshape(1, -1))

    # Print the optimized input parameters
    print("\nOptimized input parameters to get the desired output:")
    for param_name, value in zip(parameter_names, X_optimized[0]):
        print(f"{param_name}: {value:.4f}")
else:
    print("Optimization failed:", result.message)
