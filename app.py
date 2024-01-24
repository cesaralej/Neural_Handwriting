# import required packages
import streamlit as st
from PIL import Image
import numpy as np
import pickle
from streamlit_drawable_canvas import st_canvas

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

st.set_page_config(page_title="Hand Drawing and Neural Networks", page_icon=":pencil:")

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
    return A2

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
    f"<h1 style='font-size: 36px; text-align: center;'>Hand Drawing and Neural Networks</h1>",
    unsafe_allow_html=True,
)

# Introductory Message
st.markdown(
    f"<p style='font-size: 20px; text-align: center;'>Draw a number and our neural network will guess the number!</p>",
    unsafe_allow_html=True,
)

# Instructions on how to use the app

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
    print('new image upload')
    for array in image:
        line = ''
        for item in array:
            item = str(item).rjust(4)
            line = line+item
        #print(line)
    # Use the make_predictions function
    prediction = show_prediction(image, W1, b1, W2, b2)
    st.write("Prediction:", prediction)

canvas_result = st_canvas(
    #fill_color="rgba(0, 0, 0, 0)",  # Fixed fill color with some opacity
    stroke_width=6,
    stroke_color='#FFFFFF',
    background_color="#000000",
    height=150,
    width=150,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
    image = Image.fromarray(canvas_result.image_data).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    print('new image canvas')
    for array in image:
        line = ''
        for item in array:
            item = str(item).rjust(4)
            line = line+item
        #print(line)

    if image.any() != 0:
        prediction = show_prediction(image, W1, b1, W2, b2)
        #st.write("Prediction:", prediction)
        zipped_list = list(zip(list(range(len(prediction))), prediction))
        zipped_list.sort(key=lambda x: x[1], reverse=True)
        st.write("Top 3 predictions:")
        for i in range(3):
            st.write(zipped_list[i][0], "with probability", zipped_list[i][1])


    #image = Image.open(image_data).convert('L')

    # Display the image
    #st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Load the image (replace 'your_image_file.png' with the actual file path)
    #image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    
    ##
    ## #/ 255.0  # Normalize pixel values to be in the range [0, 1]

    # Use the make_predictions function
    
