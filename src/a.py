import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pprint import pprint
from sklearn.model_selection import train_test_split
import random
import pickle
import joblib
import xgboost

st.set_page_config(
    page_title = "ACM AI User App"
)
st.markdown(
    """
    <style>

    [data-testid = "stAppViewContainer"]{
    background-color: white;
    }

    [data-testid = "stHeader"]{
    background-color: rgba(0,0,0,0);
    }

    [data-testid = "stToolbar"]{
    right: 2rem;
    }

    [data-testid = "stSidebar"]{
    left: 2rem;
    background-color: white;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
# Adding a side bar

st.sidebar.markdown(
    """
    <h2>The Real Team Tu at UCSD:</h2>

    <style> 
    p {outline-color: silver;}
    p.solid{outline-style:solid;}
    </style>

    <p class="solid">
        <ul>
            <li><strong>Arvin:</strong> CS major from Marshall College</li>
            <li><strong>Chuong:</strong> CS major from Marshall College</li>
            <li><strong>Max:</strong> CS major from Sixth College</li>
            <li><strong>Rebecca:</strong> Cog Sci major from Sixth 
            College</li>
            <li><strong>Rohan:</strong> ECE major from Marshall College</li>
            <li><strong>Siya:</strong> CS major from Seventh College</li>
            <li><strong>Vincent:</strong> CS major from Sixth College</li>
        </ul>
        <p>
        <strong>Our team was established in the Fall of 2022. Our focus for this project
        is implementing Machine Learning to predict credit card fraud.
        </p>
    </p>
    """,
    unsafe_allow_html=True
)

# Adding a navigational bar

selected = option_menu(
    menu_title="Main Menu",
    options=["User Application", "EDA"],
    icons=["credit-card", "bar-chart-fill"],
    menu_icon="cast",
    orientation="horizontal",
)
if selected == "User Application":
    st.title("Is Your Credit Card Fraud?")

    st.write("Do you have a credit card?")

    yes = False
    if st.button('Yes'):
        st.write('Please continue with the instructions.')
    if st.button('No'):
        st.write('Our Machine Learning Prediction:  **100% accruacy**')
        st.write('Answer:  **Not Fraud**')

    bank = st.selectbox(
        'What is your bank? *',
        ('Chase', 'US Bank', 'Citi Bank', 'Wells Fargo', 'Truist', 
        'PNC Bank', 'TD Bank', 'Goldman Sachs', 'Capital One', 
        'Bank of New York Mellon', 'State Street', 'Citizens Financial', 
        'Silicon Valley Bank', 'Fifth Third Bancorp')
    )

    age = st.slider('How old are you?*', 0, 150, 40)

    transaction_amt = st.slider("transaction amt", 0, 1000000)

    purchase_date = st.date_input(
        'When was the purchase made?',
    )

    st.write("****Please confirm your asnwers. Submit if your informations are correct.****")
    st.write('Your bank is:  ', bank)
    st.write('Your age is:  ', age)
    st.write('When the purchase was made:  ', purchase_date)

### Training Model
def load_model():
    with open('../models/XGBoost.pkl', 'rb') as file:
        data = joblib.load(file)
    return data

model = load_model()

df = pd.read_csv("../input/creditcard.csv")
df = df.rename(columns={'Class': 'Fraud'})
df['Fraud'] = df['Fraud'].astype(int)

X = df.drop(['Fraud'], axis = 1)
Y = df["Fraud"]

xData = X.values
yData = Y.values
xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData, test_size = 0.2, random_state = 42)

def load_pipeline():
    with open('../models/preprocessing_pipeline.pkl', 'rb') as f:
        pipe = joblib.load(f)
    return pipe

pipe = load_pipeline()

#trying to add in random predictions of random row to app
if st.button("Submit"):
    st.balloons()
    ind = random.randint(0, (age + transaction_amt)%df.shape[0])
    row = df.iloc[ind]
    row = row.drop('Fraud')
    row = np.expand_dims(row, axis = 0)
    row = pipe.transform(row)
    prediction = model.predict(row)
    st.text(f"prediction: {prediction}")

# if selected == "EDA":
#     st.title("EDA")




    



    