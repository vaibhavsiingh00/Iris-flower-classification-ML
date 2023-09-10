import streamlit as st
import pickle

# loading the trained model
DecTree=pickle.load(open('modelDecTreeClassifier.pkl','rb'))
RandomForest=pickle.load(open('modelRandomForestClassifier.pkl','rb'))

st.title("Welcome to IRIS Flower Classifier App")

a = float(st.number_input("sepal length in cm"))
b = float(st.number_input("sepal width in cm"))
c = float(st.number_input("petal length in cm"))
d = float(st.number_input("petal width in cm"))

#create a dropdown for the classifier you want to use
classifier = st.selectbox('Which classifier do you want to use?', ('Decision Tree', 'Random Forest'))

# Button to predict the class label and also show the image of the flower accordingly file names for images are setosa.jpg, versicolor.jpg, virginica.jpg

if st.button("Predict"):
    if classifier == 'Decision Tree':
        species = DecTree.predict([[a,b,c,d]])
        if species == 0:
            # show it in bold
            st.write("The flower is Iris-Setosa")
            # show image
            st.image('setosa.jpg')
        elif species == 1:
            st.write("The flower is Iris-Versicolor")
            st.image('versicolor.jpg')
        else:
            st.write("The flower is Iris-Virginica")
            st.image('virginica.jpg')
    else:
        species = RandomForest.predict([[a,b,c,d]])
        if species == 0:
            st.write("The flower is Iris-Setosa")
            st.image('setosa.jpg')
        elif species == 1:
            st.write("The flower is Iris-Versicolor")
            st.image('versicolor.jpg')
        else:
            st.write("The flower is Iris-Virginica")
            st.image('virginica.jpg')









