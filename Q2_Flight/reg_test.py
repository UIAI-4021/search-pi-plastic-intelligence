import numpy as np

class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Add a column of ones to the input matrix for the bias term
        X = np.column_stack((np.ones(len(X)), X))

        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            predictions = np.dot(X, self.weights) + self.bias
            errors = predictions - y

            # Update weights and bias
            self.weights -= self.learning_rate * (1/len(X)) * np.dot(X.T, errors)
            self.bias -= self.learning_rate * (1/len(X)) * np.sum(errors)

    def predict(self, X):
        # Add a column of ones to the input matrix for the bias term
        X = np.column_stack((np.ones(len(X)), X))

        # Make predictions
        predictions = np.dot(X, self.weights) + self.bias
        return predictions


# Example usage:
# Assuming X is a 2D array of features and y is a 1D array of target values
X = np.array([[2, 3], [4, 5], [6, 7]])
y = np.array([8, 14, 20])

# Create and train the model
model = MultipleLinearRegression()
model.fit(X, y)

# Make predictions
new_data = np.array([[8, 9]])
predictions = model.predict(new_data)

print("Predictions:", predictions)
