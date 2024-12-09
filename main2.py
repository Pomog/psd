import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

# Define the parameter names (corresponding to the columns in your input data)
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

# Define the input data (X) and output data (Y) (already provided)
X = np.array([
    [3.2739, 59.99, 59.4, 255, -0.4702, 0.0168, 0.0025, 0.0032],
    [2.8076, 55.57, 63.9, 240, -0.4900, 0.0201, 0.0009, -0.0005],
    [2.7847, 62.39, 61.6, 280, -0.4516, -0.0191, 0.0049, 0.0023],
    [2.5080, 51.40, 57.6, 260, -0.4353, 0.0138, 0.0006, -0.0395]
])

# Output data (Y)
Y = np.array([
    [4.55, 13.80, 26.25],
    [6.24, 17.80, 28.65],
    [5.21, 16.55, 27.10],
    [7.64, 17.55, 28.50]
])

# Desired output
Y_desired = np.array([[4.0, 15.5, 58]])

# Normalize input data using StandardScaler
scaler_X = StandardScaler()
X_normalized = scaler_X.fit_transform(X)

# Normalize output data using StandardScaler
scaler_Y = StandardScaler()
Y_normalized = scaler_Y.fit_transform(Y)

# Train the model using Linear Regression
model = LinearRegression()
model.fit(X_normalized, Y_normalized)

# Define the objective function for optimization
# The function takes input parameters and returns the error between the predicted output and Y_desired
def objective(X_input):
    # X_input will be in the normalized space, so we need to predict in normalized space first
    X_input_reshaped = X_input.reshape(1, -1)  # Reshape to match the input shape
    Y_pred_normalized = model.predict(X_input_reshaped)

    # Reverse the normalization to get the predicted output in the original scale
    Y_pred = scaler_Y.inverse_transform(Y_pred_normalized)

    # Calculate the error (using Mean Squared Error, for example)
    error = np.sum((Y_pred - Y_desired) ** 2)
    return error

# Initial guess: Use the mean of the normalized inputs as a starting point
initial_guess = np.mean(X_normalized, axis=0)

# Perform the optimization to find the best input parameters
result = minimize(objective, initial_guess, bounds=[(-2, 2)] * X.shape[1], method='L-BFGS-B')

# Check if the optimization was successful
if result.success:
    # Get the optimized input parameters (normalized)
    X_optimized_normalized = result.x

    # Reverse the normalization to get the optimized input in the original scale
    X_optimized = scaler_X.inverse_transform(X_optimized_normalized.reshape(1, -1))

    # Print the optimized input parameters
    print("Optimized input parameters to get the desired output:")
    for param_name, value in zip(parameter_names, X_optimized[0]):
        print(f"{param_name}: {value:.4f}")
else:
    print("Optimization failed:", result.message)
