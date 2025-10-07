Handwriting Recognition from Scratch

A machine learning project that converts handwritten characters into typed text — built entirely from scratch using NumPy and Pandas, without TensorFlow or PyTorch. The goal was to understand the fundamentals of neural networks by implementing every component manually, from data preprocessing to model training and evaluation.

Features
- Custom-built neural network implemented purely with NumPy
- Forward and backward propagation written from first principles
- Image preprocessing and normalization pipeline
- Interactive web interface (HTML, CSS, JavaScript) for drawing characters
- Real-time prediction and text output display
- Evaluation metrics for model accuracy and loss visualization

Tech Stack
- Python (NumPy, Pandas, Matplotlib)
- HTML, CSS, JavaScript for front-end interface
- Flask (optional) for backend integration

How It Works
- Data Preprocessing: Images are converted to grayscale, flattened, and normalized between 0 and 1.
- Model Architecture: A multilayer perceptron (MLP) is implemented manually with customizable layers, activation functions (ReLU, Sigmoid, Softmax), and loss functions.
- Training: Gradient descent is used to update weights and minimize cross-entropy loss.
- Inference: The trained model predicts the most likely letter for a given drawn image.
- Web Interface: Users can draw a character on a canvas, send it to the backend, and instantly see the recognized result.

Results
- Achieved ~87% accuracy on the EMNIST handwritten dataset of 800,000+ images
- Fully interpretable model with visualized weights and activations
- Demonstrates neural network fundamentals without any external ML libraries

Future Improvements
- Add convolutional layers for better spatial feature extraction
- Expand dataset to include digits and cursive writing
- Implement real-time continuous text recognition

