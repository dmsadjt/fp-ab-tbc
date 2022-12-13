import streamlit as st
import pandas as pd
import numpy as np
import mahotas as mh
import pickle
import matplotlib.pyplot as plt


st.title('6-AB-B')
st.header('Tuberculosis CNN Prediction Model')

IMM_SIZE = 224
data = st.file_uploader("Upload a Image")
lab = {'Normal': 0, 'Tuberculosis': 1}


def diagnosis(file):
    # Download image
    ## YOUR CODE GOES HERE##
    image = mh.imread(file)

    # Prepare image to classification
    ## YOUR CODE GOES HERE##

    if len(image.shape) > 2:
        # resize of RGB and png images
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]])
    else:
        # resize of grey images
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE])
    if len(image.shape) > 2:
        # change of colormap of images alpha chanel delete
        image = mh.colors.rgb2grey(image[:, :, :3], dtype=np.uint8)

    # Show image
    ## YOUR CODE GOES HERE##

    plt.imshow(image)
    plt.savefig("temp.png")
    st.subheader("Uploaded Image")
    st.image("temp.png")

    # Load model
    ## YOUR CODE GOES HERE##

    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    # # Normalize the data
    # ## YOUR CODE GOES HERE##

    image = np.array(image) / 255

    # # Reshape input images
    # ## YOUR CODE GOES HERE##

    image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

    # # Predict the diagnosis
    # ## YOUR CODE GOES HERE##

    predict_image = model.predict(image)
    predictions = np.argmax(predict_image, axis=1)
    print(predictions)
    predictions = predictions.reshape(1, -1)[0]

    # # Find the name of the diagnosis
    # ## YOUR CODE GOES HERE##
    res = {i for i in lab if lab[i] == predictions}
    resstr = str(res)
    results = resstr.strip("'{}")

    diag = "The diagnosis result is: " + results

    return diag


if data is not None:
    st.subheader(diagnosis(data))
