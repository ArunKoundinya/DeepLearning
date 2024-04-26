import numpy as np
import pandas as pd
import streamlit as st 
import pickle

model = pickle.load(open('https://github.com/ArunKoundinya/DeepLearning/blob/main/posts/deep-learning-project-msis/best_model_traditional.pkl', 'rb'))


def main(): 
    st.title("Sentiment Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Sentiment Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    review_combined_lemma = st.text_area('Area for textual entry')
    
if __name__=='__main__': 
    main()
