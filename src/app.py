#This is the python file for the streamlit app
import streamlit as st
import pandas as pd

header = st.container()
dataset = st.container()
timeline = st.container()
modelTraining = st.container()

with header:
    st.title('Data Science Project')
    st.text('This project looks into detecting credit card fraud...')

with dataset:
    st.header('Credit Card Fraud Dataset')
    st.text('This dataset is from Kaggle and PCA. Here is the EDA report...')

    credit_card_data = pd.read_csv("creditcard.csv")
    st.write(credit_card_data.head())

    st.subheader('Fraud (1) or Not Fraud (0) Distribution in Dataset')
    fraudOrNot= pd.DataFrame(credit_card_data['Class'].value_counts())
    st.bar_chart(fraudOrNot)

with timeline:
    st.header('Timeline of project')

    st.markdown('* **first subproject!** This subproject is ....')

with modelTraining:
    st.header('Time to train model')
    st.text('Choose which hyperparamaters you want to use')
    
    #allow user to tweak hyperparamters max_depth and n_estimators
    max_depth_col, n_estimators_col = st.columns(2)
    max_depth= max_depth_col.slider('What should be the max depth?', min_value=0, max_value=10, value=2, step=1)
    n_estimators= n_estimators_col.selectbox('What is n_estimators', options=[2, 4, 6, 8, 10], index=0)


    #attempt to import the model
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression 
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error
    import pprint
    from pprint import pprint
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_val_score

    df = pd.read_csv("creditcard.csv")
    df = df.rename(columns={'Class': 'Fraud'})     
    df['Fraud'] = df['Fraud'].astype(int)

    X = df.drop(['Fraud'], axis = 1)
    Y = df["Fraud"]

    xData = X.values
    yData = Y.values

    xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData, test_size = 0.2, random_state = 42)

    pipe = Pipeline([('standardScaler', StandardScaler()), ('quantiletransformer', QuantileTransformer()), ('xgb_model', xgb.XGBRegressor())])

    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = (cross_val_score(pipe, xTrain, yTrain, cv=cv))

    from sklearn.model_selection import GridSearchCV

    param_grid_pspp = [{
        'xgb_model__objective': ['binary:hinge'],
        'xgb_model__booster': ['gbtree'],
        'xgb_model__colsample_bytree': [0.3],# 0.5, 0.7],
        #'xgb_model__gamma': [0, 10, 100, 1000],
        'xgb_model__learning_rate': [0.1],# 0.2, 0.3],
        'xgb_model__max_depth': [max_depth],
        'xgb_model__n_estimators': [n_estimators]
        #'xgb_model__reg_alpha': [0 ,5, 10, 15],
        #'xgb_model__reg_lambda': [0, 5, 10, 15]
    }]

    grid_search_pspp = GridSearchCV(pipe, param_grid_pspp, cv=5,
                                    scoring= 'recall', verbose=2, n_jobs=-1)
    grid_search_pspp.fit(xTrain, yTrain)

    st.write("%0.9f recall with a standard deviation of %0.9f" % (scores.mean(), scores.std()))



    
