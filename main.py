import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background: linear-gradient(to top, #30cfd0 0%, #330867 100%);
}
</style>
"""





detector = MTCNN()

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))


def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()

    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos
st.markdown(page_bg_img,unsafe_allow_html=True)



st.title('Welcome to JB Face-Match app')

st.write("-> This is the AI-Based web application that matches the uploaded face image to bollywood actors ")
st.write("-> Simply upload the face image of anyone...  ")

uploaded_image = st.file_uploader('choose an image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)


        features = extract_features(os.path.join('uploads',uploaded_image.name),model,detector)



        index_pos = recommend(feature_list,features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

        col1, col2 = st.columns(2)

        with col1:
            st.header('Your Uploaded Image')
            st.image(display_image)
        with col2:
            st.header("Seems like " + predicted_actor)
            st.image(filenames[index_pos],width=200)

