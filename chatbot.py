# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 16:08:00 2020

@author: Saurav
"""

import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
st.set_page_config(page_title="souravs_sentiment_analysis_app")
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

st.markdown('<style>body{text-align:center;background-color:black;color:white;align-items:center;display:flex;flex-direction:column;}</style>', unsafe_allow_html=True)
st.title("College admission assistant")
st.markdown("@souravsing ")
st.image("https://media3.giphy.com/media/S60CrN9iMxFlyp7uM8/200w.webp?cid=ecf05e47938v1hebb1bs1m9uysov7g7iws0j52oqku2k0yua&rid=200w.webp")
st.markdown("Hi I'm Samonit")
#print("bot is live")
message = st.text_input("You can start chat below")
ints = predict_class(message)
res = get_response(ints,intents)
st.success("Bot :{}".format(res))
if st.button("Ask:"):
    ints = predict_class(message)
    res = get_response(ints,intents)
    st.success("Bot :{}".format(res))

