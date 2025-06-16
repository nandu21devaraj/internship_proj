import os
import pickle
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

def extract_features(img_dir, out_file):
    # Load pre-trained InceptionV3 model, use last pooling layer output
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    print("[INFO] InceptionV3 loaded.")

    features = {}
    files = os.listdir(img_dir)
    total = len(files)
    for idx, fname in enumerate(files):
        if not fname.lower().endswith('.jpg'):
            continue
        # Extract COCO image ID from filename:
        # Example: COCO_train2014_000000318556.jpg -> 318556
        img_id = int(fname.split('_')[-1].split('.')[0])

        img_path = os.path.join(img_dir, fname)
        img = image.load_img(img_path, target_size=(299, 299))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        feat = model.predict(arr, verbose=0).flatten()
        features[img_id] = feat

        if idx % 1000 == 0:
            print(f"[INFO] Processed {idx}/{total} images.")

    with open(out_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"[DONE] Extracted features for {len(features)} images. Saved to {out_file}.")

if __name__ == '__main__':
    # Example usage:
    # Place this script in your project root and run:
    # python extract_features.py
    extract_features('train2014', 'image_features.pkl')
