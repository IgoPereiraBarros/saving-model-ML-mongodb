# coding: utf-8

# https://medium.com/up-engineering/saving-ml-and-dl-models-in-mongodb-using-python-f0bbbae256f0

import pickle
import time
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from pymongo import MongoClient


iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

xgb = XGBClassifier()

xgb.fit(X_train, y_train)


def save_model_to_database_mongodb(model, client, db, dbconnection, model_name):
    # pickling the model
    pickled_model = pickle.dumps(model)
    
    # maving model to db
    myclient = MongoClient(client)
    
    # creating database in mongodb
    mydb = myclient[db]
    
    # creating the collection
    mycollection = mydb[dbconnection]
    
    info = mycollection.insert_one({model_name: pickled_model, 
                                    'name': model_name, 
                                    'created_time': time.time()})
    print(info.inserted_id, ' salvo com sucesso!')
    details = {
        'inserted_id': info.inserted_id,
        'model_name': model_name,
        'created_time': time.time()
    }
    return details
    

def load_model_saved_to_db(model_name, client, db, dbconnection):
    json_data = {}
    
    myclient = MongoClient(client)
    
    mydb = myclient[db]
    
    mycollection = mydb[dbconnection]
    
    data = mycollection.find({'name': model_name})
    
    for d in data:
        json_data = d
    pickled_model = json_data[model_name]
    
    return pickle.loads(pickled_model)


details = save_model_to_database_mongodb(model=xgb, 
                                        client='mongodb://user:password@localhost:27017/db',
                                        db='modelo_xgb',
                                        dbconnection='models',
                                        model_name='myxgb')



xgb = load_model_saved_to_db(model_name=details['model_name'],
                      client='mongodb://user:password@localhost:27017/db',
                      db='modelo_xgb',
                      dbconnection='models')


print(iris.target_names[xgb.predict([[7.3, 2.9, 6.3, 1.8]])])

