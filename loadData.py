import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import struct

def load_idx(filename):
    with open(filename, 'rb') as f: # Open file in binary mode ('rb')
        magic, size = struct.unpack(">II", f.read(8)) # Read first 8 bytes (big-endian integers)
        if magic == 2049:  # Labels file
            return np.frombuffer(f.read(), dtype=np.uint8) #sets labels to a np array
        elif magic == 2051:  # Images file
            rows, cols = struct.unpack(">II", f.read(8)) 
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows * cols) #deconstructs image
        else:
            raise ValueError("Invalid IDX file") #error message

# loads images and labels to respective arrays
train_images = load_idx("C:\MNIST AI\MNIST-AI\EMNIST dataset\emnist-byclass-train-images-idx3-ubyte") 
train_labels = load_idx("C:\MNIST AI\MNIST-AI\EMNIST dataset\emnist-byclass-train-labels-idx1-ubyte")
test_images = load_idx("C:\MNIST AI\MNIST-AI\EMNIST dataset\emnist-byclass-test-images-idx3-ubyte")
test_labels = load_idx("C:\MNIST AI\MNIST-AI\EMNIST dataset\emnist-byclass-test-labels-idx1-ubyte")

# Normalize images (convert pixel values from 0-255 to 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

#print(train_labels.size)

#debug check
'''print("Training data shape:", train_images.shape)  # Should be (112800, 784) for balanced
print("Test data shape:", test_images.shape)      # Should be (18800, 784)
print("First 5 labels:", train_labels[:5])        # Labels are numbers (not letters yet)'''

def load_mapping(mapping_file):
    #Loads the EMNIST label mapping from file.
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                label, ascii_code = map(int, parts)
                mapping[label] = chr(ascii_code)  # Convert ASCII to character
    return mapping

# Load the label mapping
label_mapping = load_mapping("C:\MNIST AI\MNIST-AI\EMNIST dataset\emnist-byclass-mapping.txt")

# Convert the first 5 labels
#print("First 5 labels as characters:", [label_mapping[label] for label in train_labels[:5]])

def show_image(index):
    img = train_images[index].reshape(28, 28)  # Reshape to 28x28 pixels
    img = np.rot90(img, k=-1)  # Rotate 90 degrees clockwise
    img = np.fliplr(img)  # Flip horizontally (optional, sometimes needed)
    plt.imshow(img, cmap='gray')
    
    label_char = label_mapping[train_labels[index]]  # Convert to character
    
    plt.title(f"Label: {label_char}")
    plt.show()

# Test with an image
'''for i in range(0,10):
    show_image(i)  # Show the first image with its correct label'''
