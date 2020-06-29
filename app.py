import streamlit as st
from PIL import Image
# import classify
import numpy as np
import pickle


celeb_names = {
        0:"lionel_messi",
        1:"maria_sharapova",
        2:"roger_federer",
        3:"serena_williams",
        4:"virat_kohli"
}

st.title("Pengenal Wajah Public Figure")



uploaded_file = st.file_uploader("Pilih gambar...", type="jpg")
if uploaded_file is not None:

    face_recog_model = pickle.load(open("saved_model_faris.pkl","rb"))
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar terunggah", use_column_width=True)

    st.write("")

    if st.button("predict"):
        st.write("Hasil...")
        label = face_recog_model.predict(uploaded_file)
        label = label.item()

        res = celeb_names.get(label)
        st.markdown(res)
