import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from PIL import Image


st.write("""
# Wine Quality Prediction App
""")
st.subheader("This App predicts the Quality of the Wine")


image = Image.open('wine.jpg')
st.image(image, use_column_width='auto')

df = pd.read_csv("winequality-red.csv")
st.subheader("Enteries of the Dataset")
st.write(df.head())


st.sidebar.header('User Input Parameters')

def user_inputs():
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.12,1.58 ,0.52)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.6 , 0.08)
        total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,289.0 , 46.5)
        alcohol=st.sidebar.slider('alcohol', 8.4,14.9, 10.4)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.66 )
        data = {
                'volatile_acidity': volatile_acidity,
                'chlorides': chlorides,
                'total_sulfur_dioxide':total_sulfur_dioxide,
                'alcohol':alcohol,
                'sulphates':sulphates}

        features = pd.DataFrame(data, index=[0])
        return features

data = user_inputs()

st.subheader('Parameters Input By the User')
st.write(data)

reviews=[]

for i in df['quality']:
  if i>=3 and i<=4:
    reviews.append('Bad')
  elif i>4 and i<=5:
    reviews.append('Average')
  elif i>=6 and i<=8:
    reviews.append('Good')
df['Reviews']=reviews

label_encoder = LabelEncoder()
df['Reviews'] = label_encoder.fit_transform(df['Reviews'])

X =np.array(df[['volatile acidity' , 'chlorides' , 'total sulfur dioxide' , 'alcohol' , 'sulphates']])
Y = np.array(df['Reviews'])

scaler = StandardScaler()
X= scaler.fit_transform(X)

rf=RandomForestClassifier(max_depth=16,min_samples_leaf= 2,min_samples_split= 2,n_estimators= 30)
rf.fit(X,Y)

prediction=rf.predict(data)

st.subheader("Predicting the Quality of wine")

if (prediction == 0):
	st.write("Wine is of Average Quality")
elif (prediction==1):
	st.write("Wine is of Bad Quality")
elif (prediction==2):
	st.write("Wine is of Good Quality")


