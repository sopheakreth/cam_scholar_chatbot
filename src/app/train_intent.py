# src/app/train_intent.py
import json
import pickle
import random
import numpy as np
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# --- Setup ---
BASE_DIR = os.path.dirname(__file__)
INTENTS_PATH = os.path.join(BASE_DIR, "../../data/intents.json")

with open(INTENTS_PATH) as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Prepare training data ---
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open(os.path.join(BASE_DIR, "../../src/models/words.pkl"), 'wb'))
pickle.dump(classes, open(os.path.join(BASE_DIR, "../../src/models/classes.pkl"), 'wb'))

# --- Create training data ---
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

# Split features and labels
X = training[:, :len(words)]
Y = training[:, len(words):]

# --- Split into training and testing sets ---
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- Build model ---
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# --- Train model ---
hist = model.fit(
    np.array(x_train),
    np.array(y_train),
    epochs=200,
    batch_size=5,
    validation_data=(x_test, y_test),
    verbose=1
)

# --- Evaluate accuracy ---
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Among {len(y_test):d}, Test Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save(os.path.join(BASE_DIR, "../../src/models/chatbot_model.h5"))
print("Training complete! Model saved as chatbot_model.h5")