import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()



def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)

  y=[]
  for i in text:
     if i.isalnum():
         y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if  i!='subject' and i not in stopwords.words('english') and i not in string.punctuation :
      y.append(i)

  text=y[:]
  y.clear()

  for i in text:
      y.append(ps.stem(i))

  return " ".join(y)


with open('CountVectorizer.pkl', 'rb') as f:
    cv = pickle.load(f, encoding='latin1')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f, encoding='latin1')

st.title("EMAIL SPAM DETECTION SYSTEM")

input_sms=st.text_input("Enter your text")

if st.button("Predict"):
 
#preprocess
    transformed_sms=transform_text(input_sms)

#vectorize
    vector_input=cv.transform([transformed_sms])

#predict
    result=model.predict(vector_input)[0]

#display
    if result==1:
        st.header("Not spam")

    else:
        st.header("Spam")    