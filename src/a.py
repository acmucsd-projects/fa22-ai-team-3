import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title = "ACM AI User App"
)
st.markdown(
    """
    <style>
    

    [data-testid = "stAppViewContainer"]{
    background-color: #e7fcf8;
    background-image: linear-gradient(180deg, grey, silver);
    }

    [data-testid = "stHeader"]{
    background-color: rgba(0,0,0,0);
    }

    [data-testid = "stToolbar"]{
    right: 2rem;
    }

    [data-testid = "stSidebar"]{
    left: 2rem;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
# Adding a side bar

st.sidebar.markdown(
    """
    <h2 style="color: White;">Team 3 AI at UCSD</h2>
    """,
    unsafe_allow_html=True
)

# Adding a navigational bar

selected = option_menu(
    menu_title="Main Menu",
    options=["User Application", "Data", "EDA"],
    icons=["credit-card", "file-spreadsheet", "bar-chart-fill"],
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

    purchase_date = st.date_input(
        'When was the purchase made?',
    )

    st.write("****Please confirm your asnwers. Submit if your informations are correct.****")
    st.write('Your bank is:  ', bank)
    st.write('Your age is:  ', age)
    st.write('When the purchase was made:  ', purchase_date)

    if st.button("Submit"):
        st.balloons()

# Loading Raw Data
df = pd.read_csv('creditcard.csv')
df = df.rename(columns={'Class': 'Fraud'})

fraud = df.loc[df['Fraud'] == 1]
fraud.head()

info = df.info()

fraud = df[df['Fraud'] == 1]
valid = df[df['Fraud'] == 0]
outlierFraction = len(fraud)/float(len(valid))

fraud.Amount.describe()
if selected == "Data":
    st.title("Data Visualization of Credit Card Fraud")

    st.header('Raw Data:')
    st.write(df)
    st.header('Fraud:')
    st.write(fraud)
    st.header('Valid:')
    st.write(valid)
    st.header('Outlier:')
    st.write("Outlier in decimal: ", outlierFraction)
if selected == "EDA":
    st.title("EDA")




    



    