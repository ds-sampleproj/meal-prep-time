# app.py - a flask app which returns the R2 score and optimized parameters of the chosen predictive model based on GridSearchCV.
#          model options: XGBRegressor (specified as 'xgbr') or RandomForestRegressor (specified as 'rfr')


# importing libraries
import numpy as np
import pandas as pd

from flask import Flask, Response
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


# initializing Flask
app = Flask(__name__)


# loads order.csv, restaurants.csv
def define_vars():
    global orders, restaurants
    orders = pd.read_csv('source/orders.csv')
    restaurants = pd.read_csv('source/restaurants.csv')


# processing features from loaded .csv files, segmenting into train/test data
def process_data():
    global X_train, X_test, y_train, y_test

    # merging 'orders' and 'restaurants' as single DataFrame
    df = orders.merge(restaurants,on='restaurant_id')

    # extracting minute of the day where order was acknowledged
    df['order_acknowledged_at'] = pd.to_datetime(df['order_acknowledged_at']) # formatting to datetime
    df['minute_acknowledged'] = df['order_acknowledged_at'].apply(lambda x: x.hour*60 + x.minute)
    del df['order_ready_at'], df['order_acknowledged_at'] # not used as input features

    # encoding string columns as integers using LabelEncoder
    encode = LabelEncoder()
    to_encode = ['country', 'city', 'type_of_food']

    for item in to_encode:
        df[item] = encode.fit_transform(df[item])

    # preparation time outlier removal
    threshold = df['prep_time_seconds'].quantile(0.99)
    df = df[(df['prep_time_seconds'] < threshold)]

    # assigning labels
    labels = df['prep_time_seconds'] # prep_time_seconds = output variable
    del df['prep_time_seconds'], df['restaurant_id'] # removing prep_time_seconds, restaurant_id

    # assigning features
    features = df

    # segmentation of training and testing data
    X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.3,random_state=0) # using 70%/30% Train-Test Split


# runs specified model, returns r2 score and best parameters based on GridSearchCV (model = 'xgbr' or 'rfr')
@app.route('/model/<model>', methods=['GET','POST'])
def run_model(model):
    try:
        if model == 'xgbr':
            reg = XGBRegressor()
        elif model == 'rfr':
            reg = RandomForestRegressor()

        params = [{'max_depth': [1,3],
                   'n_estimators': [100,250]}]

        gridsearch_reg = GridSearchCV(reg,
                                      param_grid=params,
                                      scoring='r2',
                                      cv=5)

        gridsearch_reg.fit(X_train, y_train)

        # best parameters based on GridSearchCV
        best_params = str(gridsearch_reg.best_params_)

        # R2 score for unseen test data
        r2_score = str(np.round(gridsearch_reg.score(X_test, y_test),2))

        # returns the model, parameter configurations, best parameters based on GridSearchCV, and r2 score based on test data
        return Response('\nmodel: '+model+'\n\nparameters tested:'+str(params)+'\nbest parameters: '+best_params+'\nR2 Score = '+r2_score+'\n\n',status=200)

    except:
        return Response('\nerror: please specify model as "xgbr" for XGBRegressor or "rfr" for RandomForestRegressor.\n\n',status=400)


if __name__ == "__main__":
    define_vars()
    process_data()
    app.run(host='0.0.0.0',port='8000',debug=True)
