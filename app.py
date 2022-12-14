import streamlit as st
# import pandas as pd
from PIL import Image
import torch
import os
from stablediffusion import run

# MODEL_DIR = '/opt/models/'
# for filename in os.listdir(MODEL_DIR):
    # if filename[-4:] == '.pth':
        # filepath = os.path.join(MODEL_DIR,filename)
# MODEL_PATH = filepath

st.title("Stable Diffusion")

user_input = st.text_input("prompt", "describe your image")

if user_input is not None:
    output = run(user_input)

    for idx, v in enumerate(output):
        image = Image.open(v[1])
        st.image(image, caption=f"Score: {v[0][0]:.2f}\n", use_column_width=True)


# uploaded_file = st.file_uploader("Choose an image...", type=['png','jpeg'])
# if uploaded_file is not None:
    # image = Image.open(uploaded_file)
    # st.image(image, caption='Uploaded Image.', use_column_width=True)
    # st.write("")
    # st.write("Classifying...")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = resnet18(3, 10)
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # tensor = imgToTensor(image)
    
    # output = model(tensor)
    # _, predicted = torch.max(output.data, 1)
    # prediction = classes[predicted]

    # st.write(prediction)
