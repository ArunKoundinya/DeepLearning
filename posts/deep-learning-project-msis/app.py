import numpy as np
import pandas as pd
import streamlit as st
import pickle
import warnings
import nltk
nltk.download('stopwords')
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('best_model_traditional.pkl', 'rb') as f:
    model_rfm = pickle.load(f)
with open('model_lstm_bi_embed.pkl', 'rb') as f:
    model_bistm = pickle.load(f)
with open('model_lstm_bi_embed_pretrained.pkl', 'rb') as f:
    model_bistm_pretrained = pickle.load(f)
with open('vocab_dict.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

cols= ['review_title','review_text']


def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text into words
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a single string
    text = ' '.join(words)
    return text

stop_words = set(stopwords.words('english')) - { 'not', 'no', 'couldn', "couldn't", "wouldn't", "shouldn't", "isn't",
                                                "aren't", "wasn't", "weren't", "don't", "doesn't", "hadn't", "hasn't",
                                                "won't", "can't", "mightn't","needn't","nor","shouldn","should've","should",
                                                "weren","wouldn","mustn't","mustn","didn't","didn","doesn","did","does","hadn",
                                                "hasn","haven't","haven","needn","shan't"}

def process_sentence(sentence):
    list1 = []
    for word in sentence.split():
        if word in vocab_dict:
            list1.append(vocab_dict[word])    
        else:
            list1.append(vocab_dict["<UNK>"])
    return list1

def format_examples(data1, vocab_dict, maxlen):
    sequences_data=data1['review_combined_lemma'].apply(process_sentence).tolist()
    padded_sequences_data = pad_sequences(sequences_data, maxlen=maxlen)
    return padded_sequences_data

def main():
    st.title("Sentiment Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Sentiment Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    model_selected = st.radio('Pick Model:', ['Random Forest','Bi-LSTM with Embedded Layer','Bi-LSTM with Pre-Trained Embedded Layer'])

    review_title = st.text_area('REVIEW TITLE')
    review_text = st.text_area('REVIEW TEXT')
    features = [[review_title,review_text]]

    data = {'review_title': str(review_title),'review_text': str(review_text)}

    df=pd.DataFrame([list(data.values())], columns=cols)

    df['review_combined'] = df['review_title'] + " " + df['review_text']
    df['review_combined_lemma'] = df['review_combined'].apply(preprocess)

    if st.button("Predict"):
        #print(data)

        if model_selected == 'Random Forest':
            prediction = model_rfm.predict(df['review_combined_lemma'].values)
        elif model_selected == "Bi-LSTM with Embedded Layer":
            X_review_combined_lemma = format_examples(df, vocab_dict, 100)
            prediction = model_bistm.predict(X_review_combined_lemma).astype(float)
        else :
            X_review_combined_lemma = format_examples(df, vocab_dict, 100)
            prediction = model_bistm_pretrained.predict(X_review_combined_lemma).astype(float)

        if float(prediction) > 0.5:
            st.success('Postive!!')
        else:
            st.success('Negative!!')

if __name__=='__main__':
    main()