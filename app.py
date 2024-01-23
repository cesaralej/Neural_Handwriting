# import required packages
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pickle

# Start by creating a venv:
# python -m venv myenv

# Activate your venv:
# source venv_name/bin/activate   (mac)
# venv_name\Scripts\activate  (windows)

# Install the required packages:
# pip install -r requirements.txt

# Run the code in the terminal:
# streamlit run Syllabus_generator.py

# Read the original syllabus

# API Request to generate the syllabus

# API Request to generate the capstone project

st.set_page_config(
    page_title="Business Strategy Syllabus",
    page_icon="🌐",
    initial_sidebar_state="expanded",
)

'''
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=2,
    stroke_color='#e00',
    background_color="#fff",
    height=150,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
'''

def ReLU(Z):
    return np.maximum(Z,0)

def derivative_ReLU(Z):
    return Z > 0

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

def init_params(size):
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1,b1,W2,b2

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1 #10, m
    A1 = ReLU(Z1) # 10,m
    Z2 = W2.dot(A1) + b2 #10,m
    A2 = softmax(Z2) #10,m
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

SCALE_FACTOR = 255

with open("trained_params.pkl","rb") as dump_file:
    W1, b1, W2, b2=pickle.load(dump_file)

def make_predictions(image, W1, b1, W2, b2):
    # Flatten the image and normalize
    vect_X = image.flatten() / SCALE_FACTOR
    vect_X = vect_X[:, None]

    Z1, A1, Z2, A2 = forward_propagation(vect_X, W1, b1, W2, b2)
    prediction = get_predictions(A2)
    return prediction

def show_prediction(image, W1, b1, W2, b2):
    prediction = make_predictions(image, W1, b1, W2, b2)
    print("Prediction: ", prediction)
    #plt.gray()
    #plt.imshow(current_image, interpolation='nearest')
    #Splt.show()
    return prediction


# Title
st.markdown(
    f"<h1 style='font-size: 36px; text-align: center;'>BIG DATA & AI IN BUSINESS STRATEGY</h1>",
    unsafe_allow_html=True,
)

# Introductory Message
st.markdown(
    f"<p style='font-size: 20px; text-align: center;'>Welcome to Your AI-Driven Learning Experience!</p>",
    unsafe_allow_html=True,
)

# Instructions on how to use the app
st.markdown(
    f"<h2 style='font-size: 28px;'>How to Use:</h2>",
    unsafe_allow_html=True,
)
st.write("1. **Configure your OpenAI API key in the sidebar.**")
st.write("2. **Input your professional information on the left.**")
st.write(
    "3. **Click on 'Generate Syllabus' to receive your personalized learning plan.**"
)

# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Load the image (replace 'your_image_file.png' with the actual file path)
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    image = np.array(image) #/ 255.0  # Normalize pixel values to be in the range [0, 1]

    # Use the make_predictions function
    prediction = show_prediction(image, W1, b1, W2, b2)
    st.write("Prediction:", prediction)
