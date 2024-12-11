import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# Define the parameter names (corresponding to the columns in the input data)
parameter_names = [
    "solvent/reagent mass",
    "temperature C",
    "catalyst amount %"
]

# Experiment parameters: solvent/reagent mass, temperature C, catalyst amount %
X = np.array([
    [3, 60, 1],
    [4, 60, 1],
    [5, 50, 1],
    [6, 50, 1],
    [7, 40, 1],
    [8, 30, 1],
    [9, 20, 2]
])

# Define the output names (corresponding to the columns in the input data)
output_names = [
    "yield %",
    "purity %"
]

# Output results: yield %, purity %
Y = np.array([
    [50],
    [55],
    [60],
    [65],
    [70],
    [80],
    [90]
])


if X.shape[0] != Y.shape[0]:
    raise ValueError("Number of samples in X and Y must be the same.")

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
print("Experiment parameters. Raw data\n", X)
print("Experiment results. Raw data\n", Y)

# Scaling the input date to have similar ranges, using StandardScaler
# Mean = 0
# Standard deviation = 1
# putting all features on a comparable scale

# Create scalers
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Standardize X and Y. Each feature column is centered around 0 (mean) and scaled to unit variance.
# fit - Computes μ and σ for each column, snd stores internally for later use
# transform - applies 𝑥′=(𝑥−𝜇)/𝜎 to each 𝑥 value
# Ensures no feature dominates others due to scale
X_normalized = scaler_X.fit_transform(X)
Y_normalized = scaler_Y.fit_transform(Y)

print("Normalized Experiment parameters:\n", X_normalized)
print("means for Experiment parameters ", scaler_X.mean_)
print("Mean of X (normalized):", np.mean(X_normalized, axis=0))
print("Std of X (normalized):", np.std(X_normalized, axis=0))

print("Normalized Experiment results:\n", Y_normalized)
print("means for Output results ", scaler_Y.mean_)
print("Mean of Y (normalized):", np.mean(Y_normalized, axis=0))
print("Std of Y (normalized):", np.std(Y_normalized, axis=0))

print("Now lets see what if we use inverse_transform")
print("inverse_transform Experiment parameters:\n", scaler_X.inverse_transform(X_normalized))
print("inverse_transform Experiment parameters:\n", scaler_Y.inverse_transform(Y_normalized))

# Train the model using Linear Regression
# Y=X⋅W+b
# X: Input features
# W: Coefficients
# b: Intercept
# The goal of training is to find W and b that minimize the residual sum of squares
# differences between actual and predicted outputs

# create an instance of the LinearRegression class from scikit-learn
model = LinearRegression()

# Calculating the weights and the intercept of the linear regression equation based the normalized data
# The method starts with some initial guesses for W and b often zeros
model.fit(X_normalized, Y_normalized)

print("Calculated W\n", model.coef_)
print("Calculated b\n", model.intercept_)


# X_input represents a single set of normalized input parameters
# serves as the "feedback loop" for the optimizer - how well the current parameters perform
def objective(X_input):
    # X_input will be in the normalized space, so we need to predict in normalized space first
    X_input_reshaped = X_input.reshape(1, -1)  # Reshape to match the input shape
    Y_pred_normalized = model.predict(X_input_reshaped)

    # Reverse the normalization to get the predicted output in the original scale
    Y_pred = scaler_Y.inverse_transform(Y_pred_normalized)

    # Calculate the error
    error = np.mean((Y_pred - Y_desired) ** 2)
    print("error: ", error)
    return error


# Desired target in the original scale
Y_desired = np.array([[90]])

if Y_desired.shape[1] != Y.shape[1]:
    raise ValueError("Y_desired dimensions must match Y's output dimensions.")

# Initial guess in the normalized scale
# corresponds to normalized values at the midpoint of the scaled range
# domain knowledge or data insights should be used as the initial guess
initial_guess = np.mean(X_normalized, axis=0)


# Optimization is the process of adjusting input parameters (X_input) to minimize a target error.
# The steps include:
# initial Guess
# evaluation - call objective() to calculate the error for the current X_input
# adjustment
# convergence
# L-BFGS-B - numerical optimization algorithm
# bounds - valid ranges for each parameter in normalized space
#     "solvent/reagent mass" - 0, 1
#     "temperature C" - 1, 1
#     "catalyst amount %" - 0, 1
# ftol - threshold controls how small the objective function's
#   changes need to be between iterations before the optimizer stops
result = minimize(
    objective,
    initial_guess,
    bounds=[(-2, 2)] * X.shape[1],
    method='L-BFGS-B',
    options={'ftol': 1e-4, 'maxiter': 1000}
)

if result.success:
    # Output
    # result.x contains the optimal input parameters found by the optimizer in normalized space as a 1D array
    optimal_X_normalized = result.x

    # inverse_transform of StandardScaler expects a 2D array as input.
    # that's why reshape used
    optimal_X = scaler_X.inverse_transform(optimal_X_normalized.reshape(1, -1))

    print(result.message)

    # Prints the desired output:
    print("\nDesired target:")
    for output_name, value in zip(output_names, Y_desired[0]):
        print(f"  {output_name}: {value:.2f}")

    # Print the optimized input parameters
    print("\nOptimized input parameters to get the desired output:")
    for param_name, value in zip(parameter_names, optimal_X[0]):
        print(f"  {param_name}: {value:.2f}")

else:
    print("Optimization failed:", result.message)
