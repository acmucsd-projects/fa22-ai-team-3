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
        background-color: rgba(255,182,198,1);
    }

    [data-testid = "stHeader"]{
    background-color: rgba(0,0,0,0);
    }

    [data-testid = "stToolbar"]{
        right: 2rem;
    }

    [data-testid = "stSidebar"]{
        background-color: rgba(110,12,37,1);
        left: 2rem;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
# Adding a side bar

st.sidebar.markdown(
    """
    <h2 style="color:silver">The Real Team Tu at UCSD:</h2>

    <style> 
    p {outline-color: silver;}
    p.solid{outline-style:solid;}
    </style>

    <p class="solid">
        <ul >
            <li style="color:silver; font-family: Lucida Handwriting, cursive;">
            <strong style="color: gainsboro">Arvin:</strong> CS major from 
            Marshall College</li>
            <li style="color:silver; font-family: Lucida Handwriting, cursive;">
            <strong style="color: gainsboro">Chuong:</strong> CS major from 
            Marshall College</li>
            <li style="color:silver; font-family: Lucida Handwriting, cursive;">
            <strong style="color: gainsboro ">Max:</strong> CS major from Sixth 
            College</li>
            <li style="color:silver; font-family: Lucida Handwriting, cursive;">
            <strong style="color: gainsboro">Rebecca:</strong> Cog Sci major 
            from Sixth College</li>
            <li style="color:silver; font-family: Lucida Handwriting, cursive;">
            <strong style="color: gainsboro">Rohan:</strong> ECE major from 
            Marshall College</li>
            <li style="color:silver; font-family: Lucida Handwriting, cursive;">
            <strong style="color: gainsboro">Siya:</strong> CS major from 
            Seventh College</li>
            <li style="color:silver; font-family: Lucida Handwriting, cursive;">
            <strong style="color: gainsboro">Vincent:</strong> CS major from 
            Sixth College</li>
        </ul>
        <p style="color:silver">
        <strong>Our team was established in the Fall of 2022. Our focus for this project
        is implementing Machine Learning to predict credit card fraud.
        </p>
    </p>

    <br><br><br>
    <h2 style="color:silver">Useful Resources on Google:</h2>

    <p class="solid">
        <ul>
            <li style="color:silver"><a style="color: gainsboro; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code">
            <strong>500 AI Machine Learning Deep Learning Computer Projects</a></li>
            <li style="color:silver"><a style="color: gainsboro; text-decoration:none; font-size: 14px; line-height: 2"
            href="https://docs.google.com/document/d/1tLHnR9rI1fBc-b1BndALWmgHrIKGGWuHBzn-bqoyRbQ/edit?usp=sharing">
            <strong>Credit Card Fraud ML Project Overview</a></li>
            <li style="color:silver"><a style="color: gainsboro; text-decoration:none; font-size: 14px; line-height: 2"
            href="https://www.youtube.com/watch?v=vmEHCJofslg&start=1">
            <strong>Python Pandas Data Science</a></li>
            <li style="color:silver"><a style="color: gainsboro; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://stackoverflow.com/questions/70967042/matplotlib-not-working-in-macos-shows-error-i-tried-to-reinstall-everything-bu">
            <strong>Needing Help Installing MatplotLib Not Working in macOS</a></li>
            <li style="color:silver"><a style="color: gainsboro; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html">
            <strong>Demotrations of Multi-Metric Evaluations on Cross_Val_Score and Gri_Search</a></li>
            <li style="color:silver"><a style="color: gainsboro; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://www.youtube.com/watch?v=xl0N7tHiwlw">
            <strong>Building a Machine Learning Web App from Scratch using Streamlit</a></li>
            <li style="color:silver"><a style="color: gainsboro; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://stackoverflow.com/questions/43691380/how-to-save-load-xgboost-model">
            <strong>How to Save & Load xgboost Model</a></li>
            <li style="color:silver"><a style="color: gainsboro; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html">
            <strong>How to Create a Web Application with Fraud Detection Handbook</a></li>
            <li style="color:silver"><a style="color: gainsboro; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://www.youtube.com/watch?v=0Hhqf8L-b_0">
            <strong>How to Install Anaconda on MacOS</a></li>
            <li style="color:silver"><a style="color: gainsboro; text-decoration:none; font-size: 14px;" 
            href="https://www.dev2qa.com/how-to-fix-the-python-pillow-error-importerror-dll-load-failed-while-importing-_imaging-the-specified-module-could-not-be-found/">
            <strong>How to Fix the Python Pillow Error ImportError</a></li>
        </ul>
        <p style="color:silver">
        <strong>Above are some of the resources we found extremely useful during 
        our journey of training and building our model to predict credit card frauds. 
        </p><br><br><br>
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

if selected == "EDA":
    st.title("EDA")




    



    