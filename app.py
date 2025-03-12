import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from dataProcessing import forwardPropagation

st.markdown(
    """
    <style>
        /* Set the entire page background */
        .stApp {
            background-color: #0078D7 !important;  /* Soft blue */
        }

        /* Remove dark box effect */
        .block-container {
            background-color: transparent !important;
        }

        /* Make sidebar (if any) match the main background */
        .css-1lcbmhc {
            background-color: transparent !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom preprocessing to match EMNIST format
def preprocess_image(img):
    # Convert to grayscale and invert colors
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = 255 - img  # Invert colors (EMNIST uses white-on-black)
    
    # Resize and normalize
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # EMNIST-specific transformations
    #img = np.transpose(img)  # Transpose width/height
    #img = np.fliplr(img)     # Flip horizontally
    img = img.reshape(784, 1) / 255.0  # Flatten and normalize
    
    return img

# Label mapping for EMNIST ByClass dataset
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
st.write("Draw a character below (black on white):")

# Create drawing canvas
canvas = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    width=400,
    height=400,
    drawing_mode="freedraw",
    key="canvas",
)

# Load model
@st.cache_resource
def load_model():
    W1 = np.load('new_model_params/W1.npy')
    b1 = np.load('new_model_params/b1.npy')
    W2 = np.load('new_model_params/W2.npy')
    b2 = np.load('new_model_params/b2.npy')
    return W1, b1, W2, b2

W1, b1, W2, b2 = load_model()

if st.button("Predict"):
    if canvas.image_data is not None:
        # Preprocess the image
        processed_img = preprocess_image(canvas.image_data)
        
        # Show preprocessed image
        #st.image(processed_img.reshape(28, 28), caption="Preprocessed Image", width=100)
        
        # Make prediction
        _, _, _, A2 = forwardPropagation(W1, b1, W2, b2, processed_img)
        prediction = np.argmax(A2, axis=0)[0]
        confidence = A2[prediction][0]
        
        #st.success(f"Prediction: **{label_map[prediction]}** (Confidence: {confidence*100:.1f}%)")
        st.success(f"Prediction: **{label_map[prediction]}**")
    else:
        st.error("Please draw something first!")