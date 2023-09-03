import numpy as np
def cross_entropy_cost(A,Y):
    m=A.shape[1]
    cost = (Y.dot(np.log((A.T))) + (1 - Y).dot(np.log((1 - A).T))) * (-1 / m)
def compute_cost_with_regularization(A, Y, parameters, lambd):
    m=A.shape[1]
    entropy_cost=cross_entropy_cost(A,Y)

    L = len(parameters) // 2
    l2_cost = 0
    for i in range(1, L + 1):
        l2_cost += np.squeeze(np.sum(np.square(parameters["W" + str(i)]), keepdims=True))
    l2_cost = l2_cost * (lambd / (2 * m))
    cost = cross_entropy_cost + l2_cost
    return cost


def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T) + (lambd / m) * (W3)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) + (lambd / m) * (W2)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T) + ((lambd / m) * W1)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients