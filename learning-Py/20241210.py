import numpy as np
import pandas as pd
import pca
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score

# Define the parameter names (corresponding to the columns in the input data)
parameter_names = [
    "solvent/reagent mass",
    "temperature C",
    "catalyst amount %"
]

# Experiment parameters: solvent/reagent mass, temperature C, catalyst amount %
X = np.array([
    [3, 60, 10],
    [4, 60, 7],
    [5, 50, 6],
    [6, 50, 5],
    [7, 40, 4],
    [8, 30, 3],
    [9, 20, 1]
])

df = pd.DataFrame(X, columns=parameter_names)
correlation_matrix = df.corr()
print("If correlation values (≥0.8) are high between parameters, the model may suffer from multi-collinearity.")
print(correlation_matrix)

# Principal Component Analysis
pca = PCA(n_components=3)
# Sensitive to outliers, they can distort the variance.
X_pca = pca.fit_transform(X)

print("Explained variance:", pca.explained_variance_)
print("X_PCA:", X_pca)

num_components = sum(pca.explained_variance_ > 1)
print(f"Number of components selected by Kaiser Criterion: {num_components}")

plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue (Variance)')
plt.title('Scree Plot')
plt.axvline(x=3, color='r', linestyle='--')  # Example: 3 components
plt.show()

# Plot cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Choosing the Number of Principal Components')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()

# Reconstruct dataset using only the two principal components (PC1, PC2)
X_pca_reduced = X_pca[:, :2]  # Two principal component
print("X_pca_reduced: ", X_pca_reduced)
X_reconstructed = pca.inverse_transform(np.hstack((X_pca_reduced, np.zeros((X_pca.shape[0], 1)))))

# Compare original and reconstructed datasets
df_reconstructed = pd.DataFrame(X_reconstructed, columns=parameter_names)
print("Reconstructed dataset:")
print(df_reconstructed)

# Calculate reconstruction error (difference between original and reconstructed data)
reconstruction_error = np.abs(X - X_reconstructed)
print("Reconstruction error:")
print(pd.DataFrame(reconstruction_error, columns=parameter_names))

plt.figure(figsize=(8, 6))
for i, param in enumerate(parameter_names):
    plt.scatter(df[param], df_reconstructed[param], label=param)
plt.plot([X.min(), X.max()], [X.min(), X.max()], 'r--')  # Reference line
plt.xlabel('Original Values')
plt.ylabel('Reconstructed Values')
plt.title('Original vs Reconstructed Data')
plt.legend()
plt.show()

# Define the output names (corresponding to the columns in the input data)
output_names = [
    "yield %"
]

# Output results: yield %, purity %
Y = np.array([[50],
              [55],
              [60],
              [90],  # Local maximum
              [70],
              [80],
              [90]])

if X.shape[0] != Y.shape[0]:
    raise ValueError("Number of samples in X and Y must be the same.")

print("Shape of X:", X.shape)
print("Experiment parameters. Raw data\n", X)

print("Shape of Y:", Y.shape)
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
X_normalized = scaler_X.fit_transform(X_pca_reduced)
Y_normalized = scaler_Y.fit_transform(Y)

print("Normalized Experiment parameters PCA reduced:\n", X_normalized)
print("means for Experiment parameters ", scaler_X.mean_)
print("Mean of X (normalized):", np.mean(X_normalized, axis=0))
print("Std of X (normalized):", np.std(X_normalized, axis=0))

print("Normalized Experiment results:\n", Y_normalized)
print("means for Output results ", scaler_Y.mean_)
print("Mean of Y (normalized):", np.mean(Y_normalized, axis=0))
print("Std of Y (normalized):", np.std(Y_normalized, axis=0))

print("Now lets see what if we use inverse_transform")
X_pca_reconstructed = scaler_X.inverse_transform(X_normalized)
X_reconstructed = pca.inverse_transform(np.hstack((X_pca_reconstructed, np.zeros((X_pca.shape[0], 1)))))
print("inverse_transform Experiment parameters:\n", X_reconstructed)
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

# Accuracy of the model
Y_pred_normalized = model.predict(X_normalized)
Y_pred = scaler_Y.inverse_transform(Y_pred_normalized)
print("Predicted results:")
print(Y_pred)

mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)
print(f"MSE: {mse}, R^2: {r2}")

residuals = Y - Y_pred
plt.scatter(Y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Linear model Residuals plot")
plt.show()

for i, param_name in enumerate(parameter_names):
    label_text = param_name + " Linear regression model"
    plt.scatter(X[:, i], Y, label=label_text)
    plt.plot(X[:, i], Y_pred, color='red')
    plt.xlabel(param_name)
    plt.ylabel("Result")
    plt.title(f"Dependency {param_name}")
    plt.legend()
    plt.show()

poly_model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
poly_model.fit(X_normalized, Y_normalized)

Y_pred_poly_normalized = poly_model.predict(X_normalized)
Y_pred_poly = scaler_Y.inverse_transform(Y_pred_poly_normalized)
print("Predicted results Poly model:")
print(Y_pred_poly)

poly_mse = mean_squared_error(Y, Y_pred_poly)
poly_r2 = r2_score(Y, Y_pred_poly)

print(f"POLY MSE: {poly_mse}, POLY R^2: {poly_r2}")
for i, param_name in enumerate(parameter_names):
    label_text = param_name + " Polynomial regression model"
    plt.scatter(X[:, i], Y, label=label_text)
    plt.plot(X[:, i], Y_pred_poly, color='red')
    plt.xlabel(param_name)
    plt.ylabel("Result")
    plt.title(f"Dependency {param_name}")
    plt.legend()
    plt.show()


# X_input represents a single set of normalized input parameters
# serves as the "feedback loop" for the optimizer - how well the current parameters perform
def objective(X_input):
    print("X_input: ", X_input)
    # X_input will be in the normalized space, so we need to predict in normalized space first
    Y_pred_normalized_ob = poly_model.predict(X_input.reshape(1, -1))

    # Reverse the normalization to get the predicted output in the original scale
    Y_pred_ob = scaler_Y.inverse_transform(Y_pred_normalized_ob.reshape(1, -1))

    print("Y_pred_ob:\n", Y_pred_ob)
    print("Y_desired:\n", Y_desired)

    # Calculate the error
    error = np.mean((Y_pred_ob - Y_desired) ** 2)
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
print("initial_guess: ", initial_guess)

# Optimization is the process of adjusting input parameters (X_input) to minimize a target error.
# The steps include:
# initial Guess
# evaluation - call objective() to calculate the error for the current X_input
# adjustment
# convergence
# L-BFGS-B - numerical optimization algorithm
# ftol - threshold controls how small the objective function's
#   changes need to be between iterations before the optimizer stops
result = minimize(
    objective,
    initial_guess,
    method='L-BFGS-B',
    options={'ftol': 1e-1, 'maxiter': 1000}
)

if result.success:
    # Output
    print("Optimization successful!")
    # the optimal input parameters found by the optimizer in normalized space
    optimal_X_normalized = result.x

    # Convert the optimal normalized parameters back to the original scale
    optimal_X_pca_reduced = scaler_X.inverse_transform(optimal_X_normalized.reshape(1, -1))

    # Use PCA to reconstruct the original scale from PCA-reduced values
    optimal_X_original = pca.inverse_transform(np.hstack((optimal_X_pca_reduced, np.zeros((1, 1)))))

    print("Optimal parameters (original scale):", optimal_X_original)

    # Predict the corresponding yield for the optimal parameters
    optimal_Y_pred_normalized = poly_model.predict(optimal_X_normalized.reshape(1, -1))
    optimal_Y_pred = scaler_Y.inverse_transform(optimal_Y_pred_normalized.reshape(1, -1))

    print("Predicted output for optimal parameters:", optimal_Y_pred)
else:
    print("Optimization failed:", result.message)

