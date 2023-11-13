import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

def data_preprocessor():
    data = pd.read_csv('Flight_Price_Dataset_Q2.csv')
    data['departure_time'] = data['departure_time'].replace(['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'] ,
                                                        [1, 2, 3, 4, 5, 6])

    data['stops'] = data['stops'].replace(['zero', 'one', 'two_or_more'],
                                      [1, 2, 3])

    data['arrival_time'] = data['arrival_time'].replace(['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'] ,
                                                        [1, 2, 3, 4, 5, 6])

    data['class'] = data['class'].replace(['Economy', 'Business'],
                                      [1, 2])

    data = data.apply(normalize_column, axis=0)

    return data

data = data_preprocessor()
x = np.array(data.iloc[:, 0:6])
y = np.array(data.iloc[:, [6]])
y = y.reshape(1, len(y))[0]

# Combine X and Y into a single array to shuffle them together
data = np.column_stack((x, y))

# Set a seed for reproducibility
np.random.seed(42)

# Shuffle the data
np.random.shuffle(data)

# Calculate the index to split the data into training and testing sets
split_index = int(0.8 * len(data))

# Split the data
train_data, test_data = data[:split_index, :], data[split_index:, :]

# Separate examples and targets for training and testing sets
x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_test, y_test = test_data[:, :-1], test_data[:, -1]

def mse(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i]) ** 2  # scalar
    cost = cost / (2 * m)  # scalar
    return cost

def mae(X, y, w, b):

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost = cost + abs(f_wb_i - y[i]) # scalar
    cost = cost / m  # scalar
    return cost



class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient Descent
        for i in range(self.n_iterations):

            predictions = np.dot(X, self.weights) + self.bias
            errors = predictions - y

            # Update weights and bias
            self.weights -= self.learning_rate * (1/len(X)) * np.dot(X.T, errors)
            self.bias -= self.learning_rate * (1/len(X)) * np.sum(errors)

            if i%200 == 0 :
                print('Epoch : ', i)

    def predict(self, X):

        # Make predictions
        predictions = np.dot(X, self.weights) + self.bias
        return predictions


# Example usage:
# Assuming X is a 2D array of features and y is a 1D array of target values
X = x_train
y = y_train

# Create and train the model
model = MultipleLinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(x_train)
print('\n')

print('R2 Score train : ', r2_score(y_train, predictions))
print('TRAIN MSE : ', mse(x_train, y_train, model.weights, model.bias))
print('TRAIN MAE : ', mae(x_train, y_train, model.weights, model.bias))
print('TRAIN RMSE : ', (mse(x_train, y_train, model.weights, model.bias))**0.5)

predictions = model.predict(x_test)
print('\n')
print('R2 Score test : ', r2_score(y_test, predictions))
print('TEST MSE : ', mse(x_test, y_test, model.weights, model.bias))
print('TEST MAE : ', mae(x_test, y_test, model.weights, model.bias))
print('TEST RMSE : ', (mse(x_test, y_test, model.weights, model.bias))**0.5)



