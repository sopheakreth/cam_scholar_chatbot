# src/app/chatbot.py
import random
import json
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import os

from recommender import recommend_universities

lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.dirname(__file__)
INTENTS_PATH = os.path.join(BASE_DIR, "../../data/intents.json")
SCHOLARSHIP_PATH = os.path.join(BASE_DIR, "../../data/scholarship_training_data.csv")

with open(INTENTS_PATH) as f:
    intents = json.load(f)

with open(SCHOLARSHIP_PATH) as f:
    universities_df = pd.read_csv(f)


# Clean column names (remove spaces)
universities_df.columns = universities_df.columns.str.strip()

# Convert Majors column into list
universities_df["Majors"] = universities_df["Majors"].apply(
    lambda x: [m.strip() for m in str(x).split("|")]
)

# Convert dataframe to dictionary list (easy for chatbot)
universities = universities_df.to_dict(orient="records")

# Load pre-trained small model and supporting files
words = pickle.load(open(os.path.join(BASE_DIR, "../../src/models/words.pkl"), "rb"))
classes = pickle.load(open(os.path.join(BASE_DIR, "../../src/models/classes.pkl"), "rb"))
model = load_model("src/models/chatbot_model.h5", compile=False)

conversation_state = {"major": None, "grade": None, "location": None, "recommender": False}

# Small demo university data
# universities = [
#     {"university": "RUPP", "major": "Computer Science", "location": "Phnom Penh", "scholarship": 80},
#     {"university": "ITC", "major": "Engineering", "location": "Phnom Penh", "scholarship": 100},
#     {"university": "NUB", "major": "Business", "location": "Battambang", "scholarship": 50}
# ]


def clean_up_sentence(sentence):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for i, w in enumerate(words):
        if w in sentence_words:
            bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I do not understand."


def update_state(message):
    print("Updating conversation state...")
    msg = message.lower().strip()

    # ---- Detect Major ----
    if conversation_state["major"] is None:
        for uni in universities:
            for major in uni["Majors"]:
                if major.lower() in msg:
                    conversation_state["major"] = major
                    return

    # ---- Detect Grade ----
    if conversation_state["grade"] is None:
        grade = msg.upper()
        if grade in {"A", "B", "C", "D", "E"}:
            conversation_state["grade"] = grade
            return

    # ---- Detect Location ----
    if conversation_state["location"] is None:
        for uni in universities:
            if uni["Location"].lower() in msg:
                conversation_state["location"] = uni["Location"]
                return

def recommend_university():

    print("Recommending universities...")

    grade = conversation_state["grade"]
    major = conversation_state["major"]
    location = conversation_state["location"]

    results = recommend_universities(
        student_grade=grade,
        major=major,
        province=location
    )

    conversation_state["major"] = None
    conversation_state["grade"] = None
    conversation_state["location"] = None
    conversation_state["recommender"] = False

    return results


def chatbot_response(message):
    # Step-by-step enforced
    print(f"Current conversation state: {conversation_state}")
    if conversation_state["recommender"] is False:
        ints = predict_class(message)
        tag = ints[0]['intent']
        recommend_intents = ["ask_major", "ask_grade", "ask_location"]
        if tag in recommend_intents:
            print(f"Predicted intent: {tag}")
            conversation_state["recommender"] = True
        return [{"intent": tag}]
    else:
        if conversation_state["major"] is None:
            update_state(message)
            if conversation_state["major"] is None:
                return [{"intent": "ask_major"}]
        if conversation_state["grade"] is None:
            update_state(message)
            if conversation_state["grade"] is None:
                return [{"intent": "ask_grade"}]
        if conversation_state["location"] is None:
            update_state(message)
            if conversation_state["location"] is None:
                return [{"intent": "ask_location"}]
        return [{"intent": "show_recommendation"}]
