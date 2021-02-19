import pandas as pd
import streamlit as st

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from PIL import Image 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import metrics


st.title("NeuroKey")


st.header("Welcome to NeuroKey! NeuroKey is a python based web app that utilizes machine learning"
" and keystroke/typing datasets in order to gauge whether or not there is a presence of Parkinson's disease within an individual.")
st.header("This app is geared towards clinicians who have already completed these tests on patients and can gain an understanding "
         " based on data whether or not the patient has Parkinson's disease.")
st.header("In order to detect the presence of Parkinson's disease, you will need to input certain variables")

st.header("Variables:")
st.subheader("pID: patient IDs")
st.subheader("updrs108: Unified Parkinson’s Disease Rating Scale part III (UPDRS-III) ")
st.subheader("PD: presence of Parkinson's disease (1 = yes, 0 = no)")
st.subheader("afTap = alternating finger tapping")
st.subheader("sTap =  single key tapping")
st.subheader("nqScore = neuroQWERTY index")



st.markdown( '<style> body{background-color: powderblue;} </style>', unsafe_allow_html = True)

parkin_data = pd.read_csv('C:\BIBEK\GT_DataPD_MIT-CS2PD.csv')


st.subheader("Data table of variables:")
st.dataframe(parkin_data)

st.write("Dataset Citation: L. Giancardo, A. Sánchez-Ferro, T. Arroyo-Gallego, I. Butterworth, C. S. Mendoza, P. Montero, M. Matarazzo, J. A. Obeso, M. L. Gray, R. San José Estépar. Computer keyboard interaction as an indicator of early Parkinson's disease. Scientific Reports 6, 34468; doi: 10.1038/srep34468 (2016)")


X = parkin_data.iloc[:, 2:7].values
Y = parkin_data.iloc[:, 1].values

#split variables for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.50, random_state = 1)

#feature input from user
def get_input():
    score = st.sidebar.slider("What is your UPDRS score?", min_value = 0, max_value = 40)
    afTap = st.sidebar.slider("What is your afTap score?", min_value = 0.0, max_value = 200.0)
    sTap = st.sidebar.slider("What is your sTap score?", min_value = 0.000, max_value = 200.00)
    nqScore = st.sidebar.slider("What is your nqScore?", min_value = 0.000, max_value = 0.500)
    typing_speed = st.sidebar.slider("What is your typing speed?", min_value = 0.00, max_value = 400.00)
    
    user_data = { "score": score, "afTap": afTap, "sTap": sTap, 
             "nqScore": nqScore, "typing speed": typing_speed}

    pd_data = pd.DataFrame(user_data, index = [0])
    return pd_data

    


user_input = get_input()

st.header("Input")
st.write(user_input)

A = svm.SVC(kernel = "linear")

A.fit(X_train, Y_train)

prediction = A.predict(user_input)


    
st.header("Presence of Parkinson's:")


if prediction == 1:
    st.subheader("Parkinson's disease has been detected")
else:
    st.subheader("Parkinson's disease has not been detected")
    
    






