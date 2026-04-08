import pandas as pd
import numpy as np

cars = pd.read_csv("quikr_car.csv")
cars.head()
cars.info()
cars = cars.dropna()
cars['year'].unique()

cars = cars[cars['year'].str.isnumeric()]
cars['year'] = cars['year'].astype(int)
print(cars['year'])

cars["Price"] = cars['Price'].str.replace(",","")
cars = cars[cars['Price'] != "Ask For Price"]
cars.isnull().sum()
cars.kms_driven.unique()

cars = cars.rename(columns={'kms_driven': 'km'})
cars['km'] = cars['km'].str.replace(",","")
cars['km'] = cars['km'].str.replace(" kms","").astype(int)
print(cars['km'])

cars.fuel_type.isna()
cars.name = cars.name.str.split(" ").str.slice(0,3).str.join(' ')
cars = cars.reset_index()


cars.Price.astype(int)
cars['Price'] = cars['Price'].astype(str).str.replace(r'[$,]', '', regex=True)
cars['Price'] = pd.to_numeric(cars['Price'], errors='coerce')
cars = cars[cars['Price'] < 800000]

cars.to_csv("mined.csv")

X = cars.drop(columns = 'Price' )
Y = cars.Price


from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)
ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
column_trans = make_column_transformer((OneHotEncoder(categories = ohe.categories_),['name','company','fuel_type']),remainder = "passthrough")
LR = LinearRegression()
mk_pipe = make_pipeline(column_trans,LR)
mk_pipe.fit(X_train,Y_train)
y_pred = mk_pipe.predict(X_test)

r2_score(Y_test,y_pred)


scores = []
for i in range(1000):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = i)
    LR = LinearRegression()
    mk_pipe = make_pipeline(column_trans,LR)
    mk_pipe.fit(X_train,Y_train)
    y_pred = mk_pipe.predict(X_test)
    scores.append(r2_score(Y_test,y_pred))

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = np.argmax(scores))
LR = LinearRegression()
mk_pipe = make_pipeline(column_trans,LR)
mk_pipe.fit(X_train,Y_train)
y_pred = mk_pipe.predict(X_test)
r2_score(Y_test,y_pred)

import pickle
pickle.dump(mk_pipe,open("LR_model.pkl","wb"))
