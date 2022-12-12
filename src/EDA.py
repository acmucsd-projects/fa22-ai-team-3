import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

    
df = pd.read_csv('pages/creditcard.csv')
df = df.rename(columns={'Class': 'Fraud'})

info = df.info()

chart_select = st.sidebar.selectbox(
    'Please pick the type opf plot you want to see',
    ('Heat map', 'Box Plots', 'Scatter Plots')
)

st.title('Exploratory Data Analysis')
fig= plt.figure(figsize = (12, 9))



if chart_select == 'Heat map':
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


