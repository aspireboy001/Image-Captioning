# app.py

from flask import Flask, render_template, request, send_from_directory
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential, Model
import pickle
import os
import numpy as np

app = Flask(__name__)

# Get paths from environment variables or use default values
model_path = os.environ.get("MODEL_PATH", "models/model.h5")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "models/tokenizer.pkl")
features_path = os.environ.get("FEATURES_PATH", "models/features.pkl")
image_path = ""


max_length = 34 

# Load the model and tokenizer
caption_model = load_model(model_path)

with open(tokenizer_path, "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the features
with open(features_path, "rb") as features_file:
    features = pickle.load(features_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        uploaded_file = request.files['file']
        image_filename = uploaded_file.filename
        image_path = os.path.join("uploads", image_filename)
        uploaded_file.save(image_path)

        # Perform prediction
        # img = load_img(image_path, target_size=(224, 224))
        # img = img_to_array(img)
        # img = img / 255.

        caption = predict_caption(caption_model, image_path, tokenizer, max_length, features)

        start_token = 'startseq'
        end_token = 'endseq'

        # Remove start token if present
        if caption.startswith(start_token):
            caption = caption[len(start_token):].lstrip()

        # Remove end token if present
        if caption.endswith(end_token):
            caption = caption[:-len(end_token)].rstrip()

        return render_template('index.html', image_filename=image_filename, caption=caption)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length, features):
    image = load_img(image,target_size=(224,224))
    image = img_to_array(image)
    image = image/255.
    image = np.expand_dims(image,axis=0)
    feature = fe.predict(image, verbose=0)
    # feature = features[image]
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text

if __name__ == '__main__':
    # for prediction
    model = DenseNet201()
    fe = Model(inputs=model.input, outputs=model.layers[-2].output)
    app.run(debug=True)
