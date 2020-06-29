import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array,load_img

@st.cache(allow_output_mutation=True)

def get_model():
    # with open("saved_model_faris.pkl", "rb") as file:
    face_recog_model = pickle.load(open("saved_model_faris.pkl","rb"))
    print("Model loaded")
    return face_recog_model

def predict(image):
    loaded_model = get_model()
    image = load_img(image, target_size=(32, 32), color_mode = "grayscale")
    image = img_to_array(image)
    image = image/255.0
    image = np.reshape(image,[1,32,32,1])

    classes = loaded_model.predict_classes(image)

    return classes
