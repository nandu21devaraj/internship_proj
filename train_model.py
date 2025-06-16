# train_model.py

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tokenizer_utils import load_tokenizer, max_length

# -----------------------------
# 1. Load data
# -----------------------------
print("[INFO] Loading image features...")
with open('image_features.pkl', 'rb') as f:
    features = pickle.load(f)

print("[INFO] Loading captions...")
with open('captions.txt') as f:
    captions = f.read().strip().split('\n')

tokenizer = load_tokenizer()
vocab_size = len(tokenizer.word_index) + 1
max_len = max_length(captions)
print(f"[INFO] Vocab Size: {vocab_size} | Max Length: {max_len}")

# -----------------------------
# 2. Create data generator
# -----------------------------
def data_generator(descriptions, photos, tokenizer, max_len):
    while True:
        for desc in descriptions:
            tokens = desc.split()
            photo_id = tokens[0]
            photo = photos.get(photo_id)
            if photo is None:
                continue
            tokens = tokens[1:]
            for i in range(1, len(tokens)):
                in_seq, out_seq = tokens[:i], tokens[i]
                in_seq = tokenizer.texts_to_sequences([' '.join(in_seq)])[0]
                in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                out_seq = tokenizer.texts_to_sequences([out_seq])[0][0]
                yield (photo, in_seq), out_seq

# -----------------------------
# 3. Build model
# -----------------------------
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_len,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

print(model.summary())

# -----------------------------
# 4. Wrap generator with Dataset
# -----------------------------
batch_size = 64

def generator():
    for (photo, in_seq), out_seq in data_generator(captions, features, tokenizer, max_len):
        yield (photo, in_seq), out_seq

output_signature = (
    (
        tf.TensorSpec(shape=(2048,), dtype=tf.float32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32)
    ),
    tf.TensorSpec(shape=(), dtype=tf.int32)
)

dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
dataset = dataset.shuffle(512).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

steps_per_epoch = len(captions) // batch_size

# -----------------------------
# 5. Train with checkpoints
# -----------------------------
checkpoint = ModelCheckpoint('model.keras', monitor='loss', save_best_only=True)

print("[INFO] Starting training...")
model.fit(dataset, epochs=20, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint])

print("[INFO] Training finished.")
