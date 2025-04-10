# python3 -m streamlit run stream1.py
import streamlit as st
import pandas as pd
import numpy as np

#1) BASICS SECTION
# st.title("Hello buddy")
# st.write("How are you?")

# df = pd.DataFrame({
#     'First-Col': [1, 2, 3, 4],
#     'Second-Col': [5, 6, 7, 8]
# })

# st.write("Here is the dataframe:")
# st.write(df)  

# chart_data=pd.DataFrame(np.random.randn(20,3),columns=['a','b','c'])


# 2) WIDGET SECTION
# st.write(chart_data)
# st.line_chart(chart_data)

# st.title("widgets")


# options=['Male','Female','Others']
# name=st.text_input("Enter your name")
# age=st.slider("Select your age",1,100,25)
# gender=st.selectbox("Choose your gender",options)
# if name:
#     st.write(f"Hello {name}, your age is {age} and your gender is {gender}")

# uploaded_file=st.file_uploader("Choose a file",type="csv")


#3) ML SECTION

from sklearn.datasets import load_iris # type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore

# @st.cache_data #don't load data from library everytime
def load_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['species']=iris.target
    # st.write(iris.feature_names)
    return df, iris.target_names
    
df,target_names=load_data()
# st.write(target_names)
model=RandomForestClassifier(n_estimators=100,random_state=1)
model.fit(df.iloc[:,:-1],df['species'])

st.sidebar.title("Input features")
sepal_length=st.sidebar.slider("Sepal length(cm)",df['sepal length (cm)'].min(),df['sepal length (cm)'].max())
sepal_width=st.sidebar.slider("Sepal width(cm)",df['sepal width (cm)'].min(),df['sepal width (cm)'].max())
petal_length=st.sidebar.slider("Petal length(cm)",df['petal length (cm)'].min(),df['petal length (cm)'].max())
petal_width=st.sidebar.slider("Petal width(cm)",df['petal width (cm)'].min(),df['petal width (cm)'].max())

predictedValue=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

st.title("Prediction")
st.write("The predicted species is",target_names[predictedValue[0]])