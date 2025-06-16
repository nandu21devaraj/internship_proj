# auto_caption.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#############################################
# === 1) Encoder (InceptionV3) ===
#############################################

def build_encoder():
    """Build the InceptionV3 encoder model for feature extraction."""
    base_model = InceptionV3(weights='imagenet')
    encoder = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return encoder

def extract_feature(img_path, encoder_model):
    """Extract feature vector for a single image."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image '{img_path}' not found.")

    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    feature = encoder_model.predict(img_array, verbose=0)
    feature = np.reshape(feature, (1, 2048))
    return feature

#############################################
# === 2) Caption Generator ===
#############################################

def generate_caption(model, tokenizer, photo, max_len):
    """Generate caption for an image feature vector."""
    in_text = 'startseq'
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()[1:-1]  # remove startseq & endseq
    return ' '.join(final)

#############################################
# === 3) Main Script ===
#############################################

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python auto_caption.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]

    print("[INFO] Loading encoder...")
    encoder = build_encoder()

    print("[INFO] Extracting feature...")
    photo = extract_feature(img_path, encoder)

    print("[INFO] Loading tokenizer & model...")
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model('model.keras')  # replace with your model file name

    max_len = 52  # replace with your training max_len

    print("[INFO] Generating caption...")
    caption = generate_caption(model, tokenizer, photo, max_len)

    print(f"\n[CAPTION] {caption}\n")

    # === Optional: Display image with caption ===
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=12)
    plt.show()
