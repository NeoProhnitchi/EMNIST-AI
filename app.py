from flask import Flask, render_template, request, jsonify
import numpy as np
from dataProcessing import initParams, forwardPropagation  # Your existing functions

app = Flask(__name__)

# Load trained parameters
W1 = np.load('model_params/W1.npy')
b1 = np.load('model_params/b1.npy')
W2 = np.load('model_params/W2.npy')
b2 = np.load('model_params/b2.npy')

# EMNIST character mapping (Balanced dataset)
label_map = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get and format image data
    image_data = np.array(request.json['image']).reshape(784, 1)
    
    # Forward propagation
    _, _, _, A2 = forwardPropagation(W1, b1, W2, b2, image_data)
    prediction = np.argmax(A2, 0)[0]
    
    return jsonify({
        'character': label_map[prediction],
        'confidence': float(A2[prediction][0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)