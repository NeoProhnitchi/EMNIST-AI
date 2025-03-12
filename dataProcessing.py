import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from loadData import train_images, train_labels, show_image, test_images, test_labels
import os

num_classes = len(np.unique(train_labels))
#print(num_classes)
data = np.array(train_images)
m, n = data.shape # m = number of samples, n = number of features (784 for 28x28 images)
#print("Data shape:", data.shape, "Samples:", m, "Features:", n)

indices = np.arange(m)
np.random.shuffle(indices)
data = data[indices]
labels = train_labels[indices]
#print(labels)

# Split dataset
X_train = data[3000:m].T  # Transpose to shape (n, m-1000)
X_dev = data[0:3000].T  # Development set (for debugging)
Y_train = labels[3000:m] # Ensure labels match the training set
Y_dev = labels[0:3000] # Labels for the dev set
#print("X_train shape:", X_train.shape, "Y_train shape:", Y_train.shape)

# After splitting the dataset, test proper splitting
#print("Y_train unique labels:", np.unique(Y_train))
#print("Y_dev unique labels:", np.unique(Y_dev))
#print("Max label in Y_train:", np.max(Y_train))  # Should be 45 for 46 classes

def initParams():
    W1 = np.random.randn(1024,784) * np.sqrt(2.0 / 784)  # He init #256 neuron layer
    b1 = np.zeros((1024,1)) # Initialize biases as zeros
    W2 = np.random.randn(62,1024) * np.sqrt(2.0 / 1024)  # He init #62 neuron output layer
    b2 = np.zeros((62,1))
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

def oneHotEncode(Y, numClasses):
    Y = Y.flatten()
    oneHotY = np.zeros((numClasses, Y.size))
    oneHotY[Y, np.arange(Y.size)] = 1
    return oneHotY

def derivReLU (Z):
    return Z > 0 #turns number (Z) into bool making it only 0 or 1 which is the deriv of ReLU

def backPropagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    oneHotY = oneHotEncode(Y, numClasses=62) #the prediction
    dZ2 = A2 - oneHotY
    dW2 = (1 / m) * dZ2.dot(A1.T) + (lambdaReg / m) * W2  # L2 term
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * derivReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T) + (lambdaReg / m) * W1  # L2 term
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def getPredictions(A2):
    return np.argmax(A2, 0)

def getAccuracy(predictions, Y):
    #print(predictions, Y)
    return (np.sum(predictions == Y) / Y.size) * 100

def saveParams(W1, b1, W2, b2, folder="new_model_params"):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Save parameters to .npy files
    np.save(f"{folder}/W1.npy", W1)
    np.save(f"{folder}/b1.npy", b1)
    np.save(f"{folder}/W2.npy", W2)
    np.save(f"{folder}/b2.npy", b2)
    print("Model parameters saved!")

def loadParams(folder="new_model_params"):
    try:
        W1 = np.load(f"{folder}/W1.npy")
        b1 = np.load(f"{folder}/b1.npy")
        W2 = np.load(f"{folder}/W2.npy")
        b2 = np.load(f"{folder}/b2.npy")
        print("Model parameters loaded!")
        return W1, b1, W2, b2
    except FileNotFoundError:
        print("No saved parameters found. Initializing new model.")
        return initParams()  # Fallback to initialization
    
def compute_loss(A2, Y):
    oneHotY = oneHotEncode(Y, numClasses=62)
    return -np.sum(oneHotY * np.log(A2 + 1e-8)) / Y.size  # Cross-entropy

lambdaReg = 0.0005
decayRate = 0.001

def gradientDescent(X, Y, iterations, alpha, saveModel=True):
    initialAlpha = alpha
    W1, b1, W2, b2 = loadParams()
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    m_W1, v_W1, m_b1, v_b1 = 0, 0, 0, 0
    m_W2, v_W2, m_b2, v_b2 = 0, 0, 0, 0
    batchSize = 256
    for i in range(iterations):
        alpha = initialAlpha * (1 / (1 + decayRate * iterations))
        # Randomly sample a mini-batch
        idx = np.random.choice(X.shape[1], batchSize, replace=False)
        X_batch = X[:, idx]
        Y_batch = Y[idx]
        Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, X_batch)
        dW1, db1, dW2, db2 = backPropagation(Z1, A1, Z2, A2, W1, W2, X_batch, Y_batch)
        
        # Adam updates
        m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
        v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1 ** 2)
        m_hat_W1 = m_W1 / (1 - beta1**(i+1))
        v_hat_W1 = v_W1 / (1 - beta2**(i+1))
        W1 -= alpha * m_hat_W1 / (np.sqrt(v_hat_W1) + epsilon)

        # Adam updates
        m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
        v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2 ** 2)
        m_hat_W2 = m_W2 / (1 - beta1**(i+1))
        v_hat_W2 = v_W2 / (1 - beta2**(i+1))
        W2 -= alpha * m_hat_W2 / (np.sqrt(v_hat_W2) + epsilon)

        # Adam updates
        m_b1 = beta1 * m_b1 + (1 - beta1) * db1
        v_b1 = beta2 * v_b1 + (1 - beta2) * (db1 ** 2)
        m_hat_b1 = m_b1 / (1 - beta1**(i+1))
        v_hat_b1 = v_b1 / (1 - beta2**(i+1))
        b1 -= alpha * m_hat_b1 / (np.sqrt(v_hat_b1) + epsilon)

        # Adam updates
        m_b2 = beta1 * m_b2 + (1 - beta1) * db2
        v_b2 = beta2 * v_b2 + (1 - beta2) * (db2 ** 2)
        m_hat_b2 = m_b2 / (1 - beta1**(i+1))
        v_hat_b2 = v_b2 / (1 - beta2**(i+1))
        b2 -= alpha * m_hat_b2 / (np.sqrt(v_hat_b2) + epsilon)

        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy (%): ", getAccuracy(getPredictions(A2), Y_batch))
            print("Loss: ", compute_loss(A2, Y_batch))

            _, _, _, A2_dev = forwardPropagation(W1, b1, W2, b2, X_dev)
            dev_loss = compute_loss(A2_dev, Y_dev)
            dev_acc = getAccuracy(getPredictions(A2_dev), Y_dev)
            print(f"Validation Loss: {dev_loss:.3f}, Validation Acc: {dev_acc:.2f}%")
            print()
            #show_image(i)

            stopAns = input("stop?: [y/n]")
            if stopAns == "y":
                break
            else:
                continue
    if saveModel:
        saveParams(W1, b1, W2, b2)  # Save after training
    return W1, b1, W2, b2

#W1, b1, W2, b2 = gradientDescent(X_train, Y_train, 2001, 0.001, saveModel=True)

#print("Unique labels in dataset:", np.unique(Y_train))

#print(load_params(folder="model_params"))




#test on test images of EMNIST dataset


def testPredict(X, W1, b1, W2, b2):
    _, _, _, A2 = forwardPropagation(W1, b1, W2, b2, X)
    return np.argmax(A2, 0)

# Preprocess test data
X_test = (np.array(test_images)).T  # Shape: (784, num_test_samples)
Y_test = test_labels

# Load model
W1, b1, W2, b2 = loadParams()

# Predict
test_predictions = testPredict(X_test, W1, b1, W2, b2)

def getTestAccuracy(predictions, Y):
    #print(predictions, Y)
    return (np.sum(predictions == Y) / Y.size) * 100

# Metrics
test_accuracy = getTestAccuracy(test_predictions, Y_test)
_, _, _, A2_test = forwardPropagation(W1, b1, W2, b2, X_test)
test_loss = compute_loss(A2_test, Y_test)

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Loss: {test_loss:.3f}")

def show_test_sample(index):
    image = X_test[:, index].reshape(28, 28)
    image = np.rot90(image, k=-1)  # Rotate 90 degrees clockwise
    image = np.fliplr(image)  # Flip horizontally (optional, sometimes needed)
    plt.imshow(image, cmap='gray')
    plt.title(f"Pred: {test_predictions[index]}, True: {Y_test[index]}")
    plt.show()

# Example: Show first 5 test samples
"""for i in range(5):
    show_test_sample(i)"""

