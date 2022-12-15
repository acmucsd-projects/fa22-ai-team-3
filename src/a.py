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
import time
# from link_button import link_button

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
    menu_title=None,
    options=["User Application", "EDA"],
    icons=["credit-card", "bar-chart-fill"],
    menu_icon="cast",
    orientation="horizontal",
)
if selected == "User Application":
    tab1, tab2, tab3,= st.tabs(["Page 1", "Page 2", "Page 3"])
    complete = 0   
    decimal_num = float((100/3)/4 )
    third = 100/3
    thirdofthird = third/3
    thirdofthirdofthird = thirdofthird/3
    bank = None
    purchase_date = None

#### Page 1

    with tab1:
        st.markdown(
            """
            <br>
            <p  style="color:silver; font-family: Lucida Handwriting, cursive;
            font-size: 2.5em;"><strong style="color: rgba(110,12,37,1)">
            Please Answer the Questions Below:</p>
            """,
            unsafe_allow_html=True,
            )
        col1, col2 = st.columns(2)

### First and Last Name
        with col1:
            st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                First and Last Name*:</p>
                """,
                unsafe_allow_html=True,
            )
            first = st.text_input('First', '')

        with col2:
            st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(255,182,198,1)">Fir</p>
                """,
                unsafe_allow_html=True,
            )
            last = st.text_input('Last', '')

        if first == '' or last == '':
            st.warning('Warning. Please Fill in your first and last name.', icon="⚠️")
        
        else:
            complete += decimal_num

### Address

        col3, col4, col5, col6= st.columns([0.5, 0.25, 0.2, 0.2])

        with col3:
            st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                Address*:</p>
                """,
                unsafe_allow_html=True,
            )
            address = st.text_input('Address', '')
        
        with col4:
            st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(255,182,198,1)">Fir</p>
                """,
                unsafe_allow_html=True,
            )
            city = st.text_input('City', '')

        
        with col5:
            st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(255,182,198,1)">Fir</p>
                """,
                unsafe_allow_html=True,
            )
            state = st.text_input('State', '')
        
        with col6:
            st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(255,182,198,1)">Fir</p>
                """,
                unsafe_allow_html=True,
            )
            zip = st.text_input('Zip Code', '')

        if address == '' or city == '' or state == '' or zip == '':
            st.warning('Warning. Please Fill in your address.', icon="⚠️")
        
        else:
            complete += decimal_num

### Gender

        st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                Gender*:</p>
                """,
                unsafe_allow_html=True,
            )
        gender = st.selectbox('Gender', ['Select Your Gender','Male', 'Female', 'Transgender', 
        'Non-binary/non-conforming', 'Prefer not to respond'])

        if gender == 'Select Your Gender':
            st.warning('Warning. Please seclect your preferred choice.', icon="⚠️")
        
        else:
            complete += decimal_num

### DoB

        col7, col8, col9 = st.columns(3)

        with col7:
            st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                Date of Birth*:</p>
                """,
                unsafe_allow_html=True,
            )
            month = st.number_input('Month',0,12)
        
        with col8:
            st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(255,182,198,1)">Fir</p>
                """,
                unsafe_allow_html=True,
            )
            date = st.number_input('Date',0,31)
        
        with col9:
            st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(255,182,198,1)">Fir</p>
                """,
                unsafe_allow_html=True,
            )
            year = st.number_input('Year', 1850,2022)
                
        if month == 0 or date == 0 or year == 1850:
            st.warning('Warning. Please fill in your date of birth.', icon="⚠️")
        
        else:
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
        st.caption("You are {}% finished. Please access the next two pages at the top to complete your application.".format(round(complete,2)))

##### Page 2

    with tab2:
        st.markdown(
            """
            <br>
            <p  style="color:silver; font-family: Lucida Handwriting, cursive;
            font-size: 2.5em;"><strong style="color: rgba(110,12,37,1)">
            Please Answer the Questions Below:</p>
            """,
            unsafe_allow_html=True,
        )       
        decimal_num = 2

### Has Credit Card or not
        st.markdown(
            """
            <br>
            <p  style="color:silver; font-family: Lucida Handwriting, cursive;
            font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
            Do You Have a Credit Card*:</p>
            """,
            unsafe_allow_html=True,
        )

        card = st.selectbox('', ['Select your option', 'Yes', 'No'])

        if card == 'Select your option':
            st.warning('Warning. Please select your preferred option.', icon="⚠️")

        elif card == 'No':
            complete += third

            with st.expander("**Why having a credit card is beneficial?**"):
                st.markdown("""
                    <br>
                    <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                    font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                    According to Expreian:</p>

                    <style> 
                    p {outline-color: rgba(110,12,37,1);}
                    p.solid{outline-style:solid;}
                    </style>

                    <p class="solid">
                        <ul >
                        <br>
                            <li style="color: rgba(110,12,37,1);">
                            Credit Cards are useful tool to build credit.</li>
                            <li style="color: rgba(110,12,37,1);">
                            Credit cards give you rewards: such as cash backs or cumulated points
                            that could be used toward purchases or traveling.</li>
                            <li style="color: rgba(110,12,37,1);">
                            Credit Cards offers customers the financial cushion to help
                            afford during financial burdens like emergencies or purchasing 
                            high end properties.</li>
                            <li style="color: rgba(110,12,37,1);">
                            Credit Cards offers more secruity against fraud and theft that such
                            debit card and cash couldn't provide.</li>
                        </ul>  
                    </p>
                    <br>
                    <p style="color: rgba(110,12,37,1);"><strong>For more information please visit
                    <a href='https://www.experian.com/blogs/ask-experian/do-you-really-need-a-credit-card/'
                    style="color: rgba(110,12,37,1);">Experian</a>
                    for more thorough details on how
                    your questions could be answered.</strong></p>

                    <br>
                    <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                    font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                    Citation:</p>

                    <style> 
                    p {outline-color: rgba(110,12,37,1);}
                    p.solid{outline-style:solid;}
                    </style>

                    <p class="solid">
                        &nbsp&nbspAxelton, Karen. “Do You Really Need a Credit Card?” 
                        Experian, Experian, 17 &nbsp&nbspNov. 2022, 
                        https://www.experian.com/blogs/ask-experian/do-you-really- &nbsp&nbspneed-a-credit-card/. 
                    </p>
                """,
                unsafe_allow_html=True,
                )
    
        else:
            complete += thirdofthird

### Bank brand

            st.markdown(
                """
                <br>
                <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                Your Bank Brand*:</p>
                """,
                unsafe_allow_html=True,
            )
            bank = st.text_input('Bank Brand (Chase)','')

            if bank == '':
                bank = None
                st.warning('Warning. Please fill in the name of your bank.', icon="⚠️")
            else:
                complete += thirdofthird

### Transaction Made
            st.markdown(
                    """
                    <br>
                    <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                    font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                    Was there any transaction made with this credit card?*:</p>
                    """,
                    unsafe_allow_html=True,
                )
            option = st.selectbox('', ['Choose your option', 'Yes', 'No'])

            if option == 'Choose your option':
                st.warning('Warning. Please select your preferred option.', icon="⚠️")

            elif option == 'No':
                complete += thirdofthird

            else:
                complete += thirdofthirdofthird
### When
                st.markdown(
                        """
                        <br>
                        <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                        font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                        When was the most recent transaction made?*:</p>
                        """,
                        unsafe_allow_html=True,
                    )
                purchase_date = st.date_input('',)
                complete += thirdofthirdofthird

### How much
                st.markdown(
                        """
                        <br>
                        <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                        font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                        Transaction Amount*:</p>
                        """,
                        unsafe_allow_html=True,
                    )
                transaction_amt = st.number_input("", 0, 10000000, 0, 1000)

                if transaction_amt == 0:
                    st.warning('Warning. Please select your preferred option.', icon="⚠️")

                elif transaction_amt != 0:
                    complete += thirdofthirdofthird


        st.markdown(
            """
            <br><br>
            """,
            unsafe_allow_html=True,
        )
        # progress bar
        my_bar = st.progress(float(complete/100))
        st.caption("You are {}% finished. Please access the next page at the top to complete your application.".format(round(complete,2)))

#### Page 3

    with tab3:
        st.markdown(
            """
            <p  style="font-size: 2em; line-height:2.5"><strong style="color: rgba(110,12,37,1)">
            Please confirm your answers below:</p>
            """,
            unsafe_allow_html=True,
        )    
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

        st.markdown(
            """
            <br>
            """,
            unsafe_allow_html=True,
        )
        box10, box11 = st.columns([1,3.5])

        with box10:
            submit = st.button("Submit")

        with box11:
            complete += third
            my_bar = st.progress(float(complete/100))
            st.caption("You are {}% finished. Please click submit to finish.".format(round(complete,2)))

        if submit and complete == 100.0:
            with st.spinner("Please wait for the algorithm to be executed..."):
                time.sleep(0)

            ind = random.randint(0, (age + transaction_amt)%df.shape[0])
            row = df.iloc[ind]
            row = row.drop('Fraud')
            row = np.expand_dims(row, axis = 0)
            row = pipe.transform(row)
            prediction = model.predict(row)

            if (f"{prediction}") == '[0.]':
                st.balloons()

                st.markdown(
                    """
                        <br><br><br><br><br>

                        <p  style="color:silver; font-family: Lucida Handwriting, cursive;
                        font-size: 1.5em;"><strong style="color: rgba(110,12,37,1)">
                        &nbsp&nbsp&nbsp&nbsp&nbsp&nbspCongratulations, our AI model
                        has predicted that your credit card transaction is not fraud. To 
                        ensure that your card is safe from theft and fraudulent, please refer
                        to<a href="https://www.michigan.gov/ag/consumer-protection/consumer-alerts/consumer-alerts/shopping/credit-card-safety-keep-your-accounts-safe"
                        style="color: rgba(110,75,37,1);"> Michigan Department 
                        of Attorney General</a> for more consulting on how to protect
                        your credit card and most importantly your finances. Once again,
                        thank you for choosing our service and we wish a wonderful day.</p>
                    """,
                    unsafe_allow_html=True
                )
            
            else:
                st.warning('Warning. Please call 1-800-847-2911 ASAP', icon="⚠️")                
                st.markdown(
                    """
                        <br><br><br><br><br>

                        <p  style="color:silver; font-family: Lucida Handwriting, cursive;
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
    info = df.info()

    # chart_select = st.sidebar.selectbox(
    #     'Please pick the type opf plot you want to see',
    #     ('Heat map', 'Box Plots', 'Scatter Plots')
    # )

    chart_select = option_menu(
    menu_title=None,
    options=["Heat Map", "Box Plots", "Scatter Plots"],
    icons=["grid-3x3", "grip-horizontal", "grip-vertical"],
    menu_icon="cast",
    orientation="horizontal",
    )

    st.title('Exploratory Data Analysis')
    fig= plt.figure(figsize = (12, 9))

    if chart_select == 'Heat Map':
        corrmat = df.corr()
        sns.heatmap(corrmat, vmax = .8, square = True)

        st.header('Heatmap of Correlation Matix:')
        st.pyplot(plt.show())

        plt.figure(figsize=(30,20))
        cor = df.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

        st.header('Display Pearson correlation HeatMap for all variables:')
        st.pyplot(plt.show())

    elif chart_select == 'Box Plots':
        sns.set(rc={'figure.figsize':(25,5)})
        df['Fraud'] = df['Fraud'].astype(str)
        plot = sns.boxplot(data = df, x = "Amount", y = "Fraud", showfliers = True)
        st.header('Box plot based on fraud or no fraud and amount:')
        st.pyplot(fig)

        # plot = sns.boxplot(data = df, x = "V2", y = "Fraud", showfliers = True)
        # st.header('Box plot based on fraud or no fraud and V2:')
        # st.pyplot(fig)

        # sns.boxplot(data = df, x = "V5", y = "Fraud", showfliers = True)
        # st.header('Box plot based on fraud or no fraud and V5:')
        # st.pyplot(fig2)

    elif chart_select == 'Scatter Plots':
        sns.scatterplot(x=df['V13'], y=df['V17'], hue=df['Fraud'])
        st.header('Scatter plot based on V17 and V13:')
        st.pyplot(fig)

        # sns.scatterplot(x=df['V13'], y=df['V14'], hue=df['Fraud'])
        # st.header('Scatter plot based on V13 and V14:')
        # st.pyplot(fig)

        # sns.scatterplot(x=df['V13'], y=df['V12'], hue=df['Fraud'])
        # st.header('Scatter plot based on V13 and V12:')
        # st.pyplot(fig)







        



        