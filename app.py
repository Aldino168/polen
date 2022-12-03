import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import torch


@st.cache
def load_image(img):
    im = Image.open(img)
    return im
# face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')
model = torch.hub.load('ultralytics/yolov5', 'custom', path= 'best.pt', force_reload=False)
# def detect_faces(our_image):
#     new_img = np.array(our_image.convert('RGB'))
#     img = cv2.cvtColor(new_img, 1)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # detect face
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#     # draw rectangle
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h), (0,0,255), 2)
#     return img, faces


def main():
    st.title("Enhance Image")
    st.text("Created by me")

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Please Select Activity", activities)

    if choice == "Detection":
        st.subheader('Enhance Image')

        image_file= st.file_uploader("Upload Images", type=["jpg", 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(our_image)
        
        enhance_type = st.sidebar.radio("Enchance Type", ['Original', 'Gray-Scale', 'Contrast','Blurring', 'Brightness'])
        if enhance_type == 'Gray-Scale':
            new_image = np.array(our_image.convert("RGB"))
            img = cv2.cvtColor(new_image, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # st.write(new_image)
            st.image(gray)

        if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider("Contrast-Level", 0.5, 4.0)
            enhancer = ImageEnhance.Contrast(our_image)
            image_output= enhancer.enhance(c_rate)
            st.image(image_output)

        if enhance_type == 'Brightness':
            c_rate = st.sidebar.slider("Brightness-Level", 0.5, 4.0)
            enhancer = ImageEnhance.Brightness(our_image)
            image_output= enhancer.enhance(c_rate)
            st.image(image_output)

        if enhance_type == 'Blurring':
            new_img=np.array(our_image.convert('RGB'))
            # st.write(new_img)
            blur_rate = st.sidebar.slider("Blurring-level", 0.5, 5.0)
            img = cv2.cvtColor(new_img, 1)
            blur = cv2.GaussianBlur(img,(11,11), blur_rate)
            st.image(blur)
        # else:
        #     st.image(our_image, width=500)

        task = ['Kecambah', 'Smiles', 'Eyes']
        feature_choice = st.sidebar.selectbox('Choice Feature', task)
        if st.button('Process'):
            if feature_choice == "Kecambah":
                # frame = cv2.cvtColor(our_image)
                a=[]
                b=[]
                results = model(our_image)
                img = np.squeeze(results.render())
                labels = results.xyxyn[0][:, -1].numpy()
                st.image(img)
                for i in labels:
                    if lebels[i] == 1:
                        a.append(labels[i])
                    else:
                        b.append(labels[i])
                st.success("Found {} faces".format(np.size(a))

    elif choice == "About":
        st.subheader("About me")

if __name__ == '__main__':
    main()
