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
from PIL import Image
import random
import pickle
import joblib
import requests
from streamlit_lottie import st_lottie

# import xgboost
import time
# from link_button import link_button
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def name_intro(name, major_college):
     st.markdown("""
        <li style="color:black; font-family: Lucida Handwriting, cursive;">
        <strong style="color: black">{name}:</strong>{major_college}</li>""", 
        unsafe_allow_html=True,)

def line_break():
    st.markdown(
        """
        <br>
        <p  style="color:black; font-family: Lucida Handwriting, cursive;
        font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">&nbsp</p>
        """,
        unsafe_allow_html=True,)

def warning(input):
    st.warning(input, icon="⚠️")

def instruction():
    st.markdown(
        """
        <br>
        <p  style="color:black; font-size: 1.5em;"><strong>
        Please Answer the Questions Below:</p>""",
        unsafe_allow_html=True,)

def confirm():
    st.markdown(
        """
        <br>
        <p  style="color:black; font-size: 1.5em;"><strong>
        Please confirm your answers below:</p>""",
        unsafe_allow_html=True,)

animate = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_wbwrgmka.json")

st.markdown("<h1 style='text-align: center; font-size: 20'>Dive Into Credit Card Fraud</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; font-size: 20'>By the Real Team Tu</h4>", unsafe_allow_html=True)
line_break()


st_lottie(
    animate,
    height = 400,
)
line_break()

row = [[ 1.00000000e+01,  1.44904378e+00, -1.17633883e+00,
         9.13859833e-01, -1.37566665e+00, -1.97138317e+00,
        -6.29152139e-01, -1.42323560e+00,  4.84558879e-02,
        -1.72040839e+00,  1.62665906e+00,  1.19964395e+00,
        -6.71439778e-01, -5.13947153e-01, -9.50450454e-02,
         2.30930409e-01,  3.19674668e-02,  2.53414716e-01,
         8.54343814e-01, -2.21365414e-01, -3.87226474e-01,
        -9.30189652e-03,  3.13894411e-01,  2.77401580e-02,
         5.00512287e-01,  2.51367359e-01, -1.29477954e-01,
         4.28498709e-02,  1.62532619e-02,  7.80000000e+00]]

# st.set_page_config(
#     page_title = "ACM AI User App"
# )
st.markdown(
    """
    <style>

    [data-testid = "stAppViewContainer"]{{
        background-color: rgba(212, 185, 150, 1);
    }}

    [data-testid = "stHeader"]{{
    background-color: rgba(0,0,0,0);
    }}

    [data-testid = "stToolbar"]{{
        right: 2rem;
    }}

    [data-testid = "stSidebar"]{{
        background-color: rgba(160, 120, 85, 1);
        left: 2rem;
    }}

    #tabs-bui3-tab-1 {{
            background-color: rgba(0, 0, 0, 0);
    }}

    #tabs-bui3-tab-2 {{
            background-color: rgba(0, 0, 0, 0);
    }}

    #tabs-bui3-tab-3 {{
            background-color: rgba(0, 0, 0, 0);
    }}
    
    </style>
    """,
    unsafe_allow_html=True
)

### Training Model
@st.cache(hash_funcs={'xgboost.sklearn.XGBRegressor': id})
def load_model():
    with open('./models/XGBoost.pkl', 'rb') as file:
        data = joblib.load(file)
    return data

model = load_model()

@st.cache(hash_funcs={'xgboost.sklearn.XGBRegressor': id})
def load_pipeline():
    with open('./models/preprocessing_pipeline.pkl', 'rb') as f:
        pipe = joblib.load(f)
    return pipe

pipe = load_pipeline()

# Adding a side bar

st.sidebar.markdown(
    """
    <h2 style="color:black">The Real Team Tu at UCSD:</h2>

    <style> 
    p {outline-color: black;}
    p.solid{outline-style:solid;}
    </style>

    <p class="solid">
        <ul >
            <li style="color:black; font-family: Lucida Handwriting, cursive;">
            <strong style="color: black">Arvin:</strong> CS major from 
            Marshall College</li>
            <li style="color:black; font-family: Lucida Handwriting, cursive;">
            <strong style="color: black">Chuong:</strong> CS major from 
            Marshall College</li>
            <li style="color:black; font-family: Lucida Handwriting, cursive;">
            <strong style="color: black ">Max:</strong> CS major from Sixth 
            College</li>
            <li style="color:black; font-family: Lucida Handwriting, cursive;">
            <strong style="color: black">Rebecca:</strong> Cog Sci major 
            from Sixth College</li>
            <li style="color:black; font-family: Lucida Handwriting, cursive;">
            <strong style="color: black">Rohan:</strong> EE major from 
            Marshall College</li>
            <li style="color:black; font-family: Lucida Handwriting, cursive;">
            <strong style="color: black">Siya:</strong> CS major from 
            Seventh College</li>
            <li style="color:black; font-family: Lucida Handwriting, cursive;">
            <strong style="color: black">Vincent:</strong> CS major from 
            Sixth College</li>
        </ul>
        <p style="color:black">
        Our team was established in the Fall of 2022. Our focus for this project
        is implementing Machine Learning to predict credit card fraud.
        </p>
    </p>

    <br><br><br>
    <h2 style="color:black">Useful Resources on Google:</h2>

    <p class="solid">
        <ul>
            <li style="color: black;"><a style="color: black; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code">
            500 AI Machine Learning Deep Learning Computer Projects</a></li>
            <li style="color: black;"><a style="color: black; text-decoration:none; font-size: 14px; line-height: 2"
            href="https://docs.google.com/document/d/1tLHnR9rI1fBc-b1BndALWmgHrIKGGWuHBzn-bqoyRbQ/edit?usp=sharing">
            Credit Card Fraud ML Project Overview</a></li>
            <li style="color: black;"><a style="color: black; text-decoration:none; font-size: 14px; line-height: 2"
            href="https://www.youtube.com/watch?v=vmEHCJofslg&start=1">
            Python Pandas Data Science</a></li>
            <li style="color: black;"><a style="color: black; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://stackoverflow.com/questions/70967042/matplotlib-not-working-in-macos-shows-error-i-tried-to-reinstall-everything-bu">
            Needing Help Installing MatplotLib Not Working in macOS</a></li>
            <li style="color: black;"><a style="color: black; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html">
            Demotrations of Multi-Metric Evaluations on Cross_Val_Score and Gri_Search</a></li>
            <li style="color: black;"><a style="color: black; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://www.youtube.com/watch?v=xl0N7tHiwlw">
            Building a Machine Learning Web App from Scratch using Streamlit</a></li>
            <li style="color: black;"><a style="color: black; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://stackoverflow.com/questions/43691380/how-to-save-load-xgboost-model">
            How to Save & Load xgboost Model</a></li>
            <li style="color: black;"><a style="color: black; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html">
            How to Create a Web Application with Fraud Detection Handbook</a></li>
            <li style="color: black;"><a style="color: black; text-decoration:none; font-size: 14px; line-height: 2" 
            href="https://www.youtube.com/watch?v=0Hhqf8L-b_0">
            How to Install Anaconda on MacOS</a></li>
            <li style="color: black;"><a style="color: black; text-decoration:none; font-size: 14px;" 
            href="https://www.dev2qa.com/how-to-fix-the-python-pillow-error-importerror-dll-load-failed-while-importing-_imaging-the-specified-module-could-not-be-found/">
            How to Fix the Python Pillow Error ImportError</a></li>
        </ul>
        <p style="color:black">
        Above are some of the resources we found extremely useful during 
        our journey of training and building our model to predict credit card frauds. 
        </p>
    </p>

    <br><br><br>
    <h2 style="color:black">Citations:</h2>

    <p class="solid">
        <ul>
            <li style="color: black; font-size: 14px;">&nbsp&nbspAxelton, Karen. “Do You Really Need a Credit Card?” Experian, Experian, 17 Nov. 2022, 
            <a style="color: black; text-decoration:none; line-height: 2"
            href="https://www.experian.com/blogs/ask-experian/do-you-really-need-a-credit-card/">
            https://www.experian.com/blogs/ask-experian/do-you-really-need-a-credit-card/.</a></li>
            <li style="color: black;font-size: 14px;">&nbsp&nbspBolger, Nathan. “Lab.® - 'See the Future First.'” Dribbble, 2021,
            <a style="color: black; text-decoration:none; line-height: 2"
            href="https://dribbble.com/shots/14881382-lab-see-the-future-first">
            https://dribbble.com/shots/14881382-lab-see-the-future-first.</a></li>
            <li style="color: black;font-size: 14px;">
            <a style="color: black; text-decoration:none; line-height: 2"
            href="https://lottiefiles.com/114317-banking">
            Priyanshi Khanna's Animation</a></li>
            <li style="color: black;font-size: 14px;">
            <a style="color: black; text-decoration:none; line-height: 2"
            href="https://lottiefiles.com/129769-thank-you-for-participating">
            Aaron Davis's Animation</a></li>
        </ul>
        <p style="color:black">
        Above are some of the cites that we would like to acknowledge the work and writings used
        in order for our project to be made possible. 
        </p>
        <p style="color:black">
        (Note: All animations used in this application are from Lottie Files
        which are free and have been permitted by creators for usage.)
        </p>
    </p><br><br><br>
    """,
    unsafe_allow_html=True
)

# Adding a navigational bar

selected = option_menu(
    menu_title=None,
    options=["User Application", "EDA"],
    icons=["credit-card", "bar-chart-fill"],
    menu_icon="cast",
    orientation="horizontal",
)
if selected == "User Application":
    tab1, tab2, tab3,= st.tabs([" Personal Info", " Bank Info ", " Confirm Info "])
    complete = 0   
    decimal_num = float((100/3)/4 )
    third = 100/3
    thirdofthird = third/3
    thirdofthirdofthird = thirdofthird/3
    bank = None
    purchase_date = None

#### Page 1

    with tab1:
        instruction()
        col1, col2 = st.columns(2)

### First and Last Name
        with col1:
            line_break()
            first = st.text_input('First*', '')

        with col2:
            line_break()
            last = st.text_input('Last*', '')

        if first != '' or last != '':
            complete += decimal_num        

### Address

        col3, col4, col5, col6= st.columns([0.5, 0.25, 0.2, 0.2])

        with col3:
            line_break()
            address = st.text_input('Address*', '')
        
        with col4:
            line_break()
            city = st.text_input('City*', '')

        with col5:
            line_break()
            state = st.text_input('State*', '')
        
        with col6:
            line_break()
            zip = st.text_input('Zip Code*', '')

        if address != '' or city != '' or state != '' or zip != '':
            complete += decimal_num

### Gender
        line_break()
        gender = st.selectbox('Gender*', ['Select Your Gender','Male', 'Female', 'Transgender', 
        'Non-binary/non-conforming', 'Prefer not to respond'])

        if gender != 'Select Your Gender':
            complete += decimal_num

### DoB

        col7, col8, col9 = st.columns(3)

        with col7:
            line_break()
            month = st.number_input('Month*',0,12)
        
        with col8:
            line_break()
            date = st.number_input('Date*',0,31)
        
        with col9:
            line_break()
            year = st.number_input('Year*', 1850,2022)
                
        if month != 0 or date != 0 or year != 1850:
             complete += decimal_num

        st.markdown(
            """
            <br><br>
            """,
            unsafe_allow_html=True,
        )
        age = 2022 - year
        # progress bar
        my_bar = st.progress(float(complete/100))
        progression = round(complete,2)
        st.caption("You are {}% finished. Please access the next two pages at the top to complete your application.".format(progression))
        if progression < 33:
            warning('Warning. Some inputs are missing.')

##### Page 2

    with tab2:
        instruction()      
        decimal_num = 2

### Has Credit Card or not
        line_break()
        card = st.selectbox('Do You Have a Credit Card*', 
                            ['Select your option', 'Yes', 'No'])

        if card == 'No':
            complete += third

            with st.expander("**Why having a credit card is beneficial?**"):
                st.markdown("""
                    <br>
                    <p  style="color:black; font-family: Lucida Handwriting, cursive;
                    font-size: 1.5em;"><strong style="color: black">
                    According to Expreian:</p>

                    <style> 
                    p {outline-color: black;}
                    p.solid{outline-style:solid;}
                    </style>

                    <p class="solid">
                        <ul >
                        <br>
                            <li style="color: black;">
                            Credit Cards are useful tool to build credit.</li>
                            <li style="color: black;">
                            Credit cards give you rewards: such as cash backs or cumulated points
                            that could be used toward purchases or traveling.</li>
                    <p>
                        <br><strong>Citation:</strong><br>
                        &nbsp&nbspAxelton, Karen. “Do You Really Need a Credit Card?” 
                        Experian, Experian, 17 &nbsp&nbspNov. 2022, 
                        https://www.experian.com/blogs/ask-experian/do-you-really- &nbsp&nbspneed-a-credit-card/. 
                    </p>
                """,
                unsafe_allow_html=True,
                )
    
        elif card == 'Yes':
            complete += thirdofthird

### Bank brand
            line_break()
            bank = st.text_input('Your Bank Brand (Chase)*','')

            if bank == '':
                bank = None
            else:
                complete += thirdofthird

### Transaction Made
            line_break()
            option = st.selectbox('Was there any transaction made with this credit card?*', 
            ['Choose your option', 'Yes', 'No'])

            if option == 'No':
                complete += thirdofthird

            elif option == 'Yes':
                complete += thirdofthirdofthird
### When        
                line_break()
                purchase_date = st.date_input('When was the most recent transaction made?*',)
                complete += thirdofthirdofthird

### How much
                line_break()
                transaction_amt = st.number_input('Transaction Amount*', 0, 10000000, 0, 1000)

                if transaction_amt != 0:
                    complete += thirdofthirdofthird

        st.markdown(
            """
            <br><br>
            """,
            unsafe_allow_html=True,
        )
        # progress bar
        my_bar = st.progress(float(complete/100))
        progression = round(complete,2)
        st.caption("You are {}% finished. Please access the next page at the top to complete your application.".format(progression))
        if progression < 66:
            warning('Warning. Some inputs are missing.')

#### Page 3

    with tab3:
        confirm()
        selected = st.checkbox("I need to make changes")

        if selected:
            box1, box2 = st.columns(2)
            with box1:
                first = st.text_input('First', first)
            with box2:
                last = st.text_input('last', last)

            box3, box4, box5, box6 = st.columns([0.5, 0.25, 0.2, 0.2])
            with box3:
                address = st.text_input('Adress', address)
            with box4:
                city = st.text_input('City', city)
            with box5:
                state = st.text_input('State', state)
            with box6:
                zip = st.text_input('Zip Code', zip)

            gender_option = ['Select Your Gender', 'Male', 'Female',
            'Transgender', 'Non-binary/non-conforming', 'Prefer not to respond']
            index = gender_option.index(gender)
            gender = st.selectbox('Gender', gender_option, index)

            box7, box8, box9 = st.columns(3)
            with box7:
                month = st.number_input('Month', 0, 12, month)
            with box8:
                date = st.number_input('Date', 0, 31, date)
            with box9:
                year = st.number_input('Year', 1850, 2022, year)
                age = 2022 - year
            
            card_option = ['Select your option', 'Yes', 'No']
            index_card = card_option.index(card)
            card = st.selectbox('Credit Card Status', card_option, index_card)

            bank = st.text_input('Bank Brand (Chase)',bank)

            purchase_option = ['Choose your option', 'Yes', 'No']
            index_purchase = purchase_option.index(option)
            option = st.selectbox('Trasaction Made with Current Card', purchase_option, index_purchase)

            transaction_amt = st.number_input("Transaction Amount", 0, 10000000, transaction_amt, 1000)

            purchase_date = st.date_input('Transaction Date', purchase_date)

        st.write('Your Name: **{} {}**'.format(first, last))
        st.write('Address: **{} {} {} {}**'.format(address, city, state, zip))
        st.write('Gender: **{}**'.format(gender))
        st.write('Date of Birth: **{}-{}-{}**'.format(year,month,date))
        st.write('Your age: **{}**'.format(age))
        st.write('Credit Card Status: **{}**'.format(card))

### Credit card Status is yes
        if card == 'Yes':
            st.write('Bank: **{}**'.format(bank))
            st.write('Transaction made with current card: **{}**'.format(option))
            if option == 'Yes':
                st.write('Transaction Amount: **${}**'.format(transaction_amt))
                st.write('Date of transaction made: **{}**'.format(purchase_date))

        st.markdown("""<br>""",unsafe_allow_html=True,)

        box10, box11 = st.columns([1,3.5])

        with box10:
            submit = st.button("Submit")

        with box11:
            complete += third
            my_bar = st.progress(float(complete/100))
            st.caption("You are {}% finished. Please click submit to finish.".format(round(complete,2)))

        if submit and complete == 100.0:
            with st.spinner("Please wait for the algorithm to be executed..."):
                time.sleep(5)

            row = pipe.transform(row)
            prediction = model.predict(row)

            if (f"{prediction}") == '[0.]':
                st.balloons()
                animate2 = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_I2vHhFyKEb.json")
                st.markdown(
                    """
                        <br><br><br><br>
                        <p  style="color:black;
                        font-size: 1.5em;"><strong style="color: black">
                        &nbsp&nbsp&nbsp&nbsp&nbsp&nbspCongratulations, our AI model
                        has predicted that your credit card transaction is not fraud. To 
                        ensure that your card is safe from theft and fraudulent, please refer
                        to<a href="https://www.michigan.gov/ag/consumer-protection/consumer-alerts/consumer-alerts/shopping/credit-card-safety-keep-your-accounts-safe"
                        style="color: rgba(110,75,37,1);"> Michigan Department 
                        of Attorney General</a> for more consulting on how to protect
                        your credit card and most importantly your finances. Once again,
                        thank you for choosing our service and we wish you a wonderful day.</p>
                    """,
                    unsafe_allow_html=True
                )
                line_break()
                st_lottie(animate2,height=300)
            
            else:
                warning('Warning. Please call 1-800-847-2911 ASAP')                
                st.markdown(
                    """
                        <br><br><br><br><br>

                        <p  style="color:black;
                        font-size: 1.5em;"><strong style="color: red">
                        &nbsp&nbsp&nbsp&nbsp&nbsp&nbspWarning, our AI model
                        has predicted that your credit card transaction is fraud. To 
                        ensure that your card is safe from theft and fraudulent, please refer
                        to<a href="https://usa.visa.com/support/consumer/security.html"
                        style="color: rgba(110,12,37,1);"> Visa Security + Fraud Prevention</a>
                        to report your case right away.</p>
                    """,
                    unsafe_allow_html=True
                )
        
        elif submit and complete != 100.0:
            st.error("Error. The application cannot be submited because some questions are not completed.", icon="🚨")


if selected == "EDA":
    chart_select = option_menu(
    menu_title=None,
    options=["Heat Map", "Box Plots", "Scatter Plots"],
    icons=["grid-3x3", "grip-horizontal", "grip-vertical"],
    menu_icon="cast",
    orientation="horizontal",
    )

    st.title('Exploratory Data Analysis')
    if chart_select == 'Heat Map':
        image = Image.open('./images/correlation.png')
        st.image(image, caption ='correlation')
    
        image1 = Image.open('./images/correlation2.png')
        st.image(image1, caption="correlation2")
 
    elif chart_select == 'Box Plots':
        image = Image.open('./images/box_plot1.png')
        st.header('Box plot based on fraud or no fraud and amount:')
        st.image(image, caption="box_plot1")

        image2 = Image.open('./images/box_plot2.png')
        st.header('Box plot based on fraud or no fraud and V2:')
        st.image(image, caption="box_plot2")

        image3 = Image.open('./images/box_plot3.png')
        st.header('Box plot based on fraud or no fraud and V5:')
        st.image(image, caption="box_plot3")
    
    elif chart_select == 'Scatter Plots':
       image = Image.open('./images/scatter_plot1.png')
       st.header('Scatter plot based on V17 and V13:')
       st.image(image, caption="scatter_plot1")
 
       image2 = Image.open('./images/scatter_plot2.png')
       st.header('Scatter plot based on V13 and V14:')
       st.image(image2, caption="scatter_plot2")
 
       image3 = Image.open('./images/scatter_plot3.png')
       st.header('Scatter plot based on V13 and V12:')
       st.image(image3, caption="scatter_plot3")