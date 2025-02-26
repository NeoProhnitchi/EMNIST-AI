import streamlit as st
import numpy as np
import cv2  # Add this import
from streamlit_drawable_canvas import st_canvas
from dataProcessing import forwardPropagation
from loadData import test_images

@st.cache_resource
def load_model():
    W1 = np.load('new_model_params/W1.npy')
    b1 = np.load('new_model_params/b1.npy')
    W2 = np.load('new_model_params/W2.npy')
    b2 = np.load('new_model_params/b2.npy')
    return W1, b1, W2, b2

W1, b1, W2, b2 = load_model()

label_map = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]

st.title("EMNIST Letter Recognition")
st.write("Draw a letter below (white on black):")

# Canvas scaled to 280x280 but internally 28x28
canvas = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas.image_data is not None:
        # Extract and downsample to 28x28
        img = cv2.resize(canvas.image_data[:, :, 0], (28, 28), interpolation=cv2.INTER_AREA)
        
        # Invert, rotate, flip
        img = 255 - img
        img = np.rot90(img, k=-1)
        img = np.fliplr(img)
        
        # Normalize and reshape
        img_processed = img.reshape(784, 1) / 255.0

        # Debug: Show preprocessed image
        st.image(img_processed.reshape(28, 28), caption="Preprocessed Image")

        # Predict
        _, _, _, A2 = forwardPropagation(W1, b1, W2, b2, img_processed)
        prediction = np.argmax(A2, axis=0)[0]
        confidence = A2[prediction][0]

        st.success(f"Prediction: **{label_map[prediction]}** (Confidence: {confidence*100:.1f}%)")
    else:
        st.error("Please draw something first!")


# Add this to your Streamlit app
if st.button("Test with EMNIST Sample"):
    # Load a test image (from your test set)
    test_image = test_images[0].reshape(784, 1) / 255.0  # Replace with real data
    _, _, _, A2 = forwardPropagation(W1, b1, W2, b2, test_image)
    prediction = np.argmax(A2, axis=0)[0]
    confidence = A2[prediction][0]
    st.success(f"Test Prediction: **{label_map[prediction]}** (Confidence: {confidence*100:.1f}%)")