
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
from tempfile import NamedTemporaryFile
from PIL import Image
import numpy as np

categories = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad', 4: 'Surprise'}

# Define input shape and latent dimension as global variables
input_shape = (128, 128, 3)
latent_dim = 2

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Prediction function which takes an image as input and returns the predicted emotion
def load_encoder(path_to_model):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    shape_before_flattening = tf.keras.backend.int_shape(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.load_weights(path_to_model, by_name=True)
    return encoder, shape_before_flattening 

def load_decoder(path_to_model, shape_before_flattening):
    decoder_input = Input(shape=(latent_dim,))
    x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
    x = Reshape(shape_before_flattening[1:])(x)
    x = Conv2DTranspose(128, (2, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (2, 2), activation='relu', padding='same', strides=(2, 2))(x)
    x = Conv2DTranspose(32, (2, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(16, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    decoder = Model(decoder_input, x, name='decoder')
    decoder.load_weights(path_to_model, by_name=True)
    return decoder

def load_vae(path_to_model):
    encoder, shape_before_flattening = load_encoder(path_to_model)  # Load encoder and get shape_before_flattening
    decoder = load_decoder(path_to_model, shape_before_flattening)  # Pass shape_before_flattening to load_decoder
    inputs = Input(shape=input_shape)
    z_mean, z_log_var = encoder(inputs)
    z = Lambda(sampling)([z_mean, z_log_var])
    outputs = decoder(z)
    vae_model = Model(inputs, outputs, name='vae')
    return vae_model

def predict_image(filename):
    # Path of the model where model is stored
    path_to_model = r'/Users/hemanthalaparthi/Downloads/my_model.h5'

    with st.spinner('Model is being loaded..'):
        # Load VAE model
        model = load_vae(path_to_model)
        print("Done!")

    # Image loading
    img_ = Image.open(filename).convert("RGB")
    img_ = img_.resize((input_shape[0], input_shape[1]))
    img_array = np.array(img_)
    x = img_array.astype('float32') / 255.0
    img_processed = np.expand_dims(x, axis=0)

    # Prediction using already loaded model
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    return index

st.title("# Face Expressions Classification")

st.set_option('deprecation.showfileUploaderEncoding', False)

buffer = st.file_uploader("Upload a JPG File", type=['jpg'])
temp_file = NamedTemporaryFile(delete=False)

if buffer is None:
    st.text("Please upload an image file")
else:
    image = Image.open(buffer)
    temp_file.write(buffer.read())
    st.image(image, use_column_width=True)
    predict = predict_image(temp_file.name)
    face = categories[predict]
    st.write("This image most likely belongs to {} - {}".format(face), font_size=30)
