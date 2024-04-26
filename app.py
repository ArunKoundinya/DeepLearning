import numpy as np
import pandas as pd
import streamlit as st 
import pickle


with open('best_model_traditional.pkl', 'rb') as f:
    loaded_model = pickle.load(f)


def main(): 
    st.title("Sentiment Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Sentiment Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    review_combined_lemma = st.text_area('Area for textual entry')

    
    if st.button("Predict"): 
        features = [[review_combined_lemma]]
        data = {'review_combined_lemma': str(review_combined_lemma)}
        #print(data)
        df=pd.DataFrame([list(data.values())], columns=cols)

        prediction = loaded_model.predict(df['review_combined_lemma'].values)

        if prediction == 1:
            st.success('Positive!!')
        else:
            st.success('Negative!!')
      
if __name__=='__main__': 
    main()
