import streamlit as st

st.set_page_config(
    page_title = "ACM AI User App"
)

st.title("Is Your Credit Card Fraud?")
st.sidebar.success("Select a page above.")

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


    



    