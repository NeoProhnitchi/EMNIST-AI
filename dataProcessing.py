import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from loadData import train_images, train_labels, show_image

num_classes = len(np.unique(train_labels))
print(num_classes)
data = np.array(train_images)
m, n = data.shape # m = number of samples, n = number of features (784 for 28x28 images)
print("Data shape:", data.shape, "Samples:", m, "Features:", n)

indices = np.arange(m)
np.random.shuffle(indices)
data = data[indices]
labels = train_labels[indices]
print(labels)

# Split dataset
X_train = data[1000:m].T  # Transpose to shape (n, m-1000)
X_dev = data[0:1000].T  # Development set (for debugging)
Y_train = labels[1000:m] # Ensure labels match the training set
Y_dev = labels[0:1000] # Labels for the dev set
print("X_train shape:", X_train.shape, "Y_train shape:", Y_train.shape)

def initParams():
    W1 = np.random.randn(256,784) * np.sqrt(2.0 / 784)  # He init #256 neuron layer
    b1 = np.zeros((256,1)) # Initialize biases as zeros
    W2 = np.random.randn(47,256) * np.sqrt(2.0 / 256)  # He init #47 neuron output layer
    b2 = np.zeros((47,1))
    return W1, b1, W2, b2

def ReLU(Z):
    #goes through each element and if element is greater than 0 return Z and if less than 0 returns 0
    return np.maximum(0, Z) 

def softmax(Z):
    Z -= np.max(Z, axis=0, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

def forwardPropagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def oneHotEncode(Y):
    Y = Y.flatten()
    oneHotY = np.zeros((Y.max() + 1, Y.size))
    oneHotY[Y, np.arange(Y.size)] = 1
    return oneHotY

def derivReLU (Z):
    return Z > 0 #turns number (Z) into bool making it only 0 or 1 which is the deriv of ReLU

def backPropagation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    oneHotY = oneHotEncode(Y) #the prediction
    dZ2 = A2 - oneHotY
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * derivReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def getPredictions(A2):
    return np.argmax(A2, 0)

def getAccuracy(predictions, Y):
    print(predictions, Y)
    return (np.sum(predictions == Y) / Y.size) * 100

def gradientDescent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = initParams()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backPropagation(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy (%): ", getAccuracy(getPredictions(A2), Y))
            show_image(i)
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradientDescent(X_train, Y_train, 500, 0.1)

print("Unique labels in dataset:", np.unique(Y_train))