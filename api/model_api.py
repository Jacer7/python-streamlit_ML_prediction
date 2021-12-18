import pickle

from fastapi import FastAPI
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from pydantic import BaseModel

df = pd.read_csv('C:/Users/Jaysor/Desktop/house.csv')
df = pd.get_dummies(df)
print(df.head())
y = df['price']
X = df.drop(['price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)  # , test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

mse_train = mean_squared_error(y_train, model.predict(X_train))
mse_test = mean_squared_error(y_test, model.predict(X_test))
model_coefs = model.coef_

y_pred = model.predict(X_train)

# Save as in pickle file

# filename = 'model.pkl'
# pickle.dump(model, open(filename, 'wb'))




# load the model from disk
# loaded_model = joblib.load('model_houses.sav')
result = model.score(X_test, y_test)
result2 = model.score(X_train, y_pred)
print("The accuracy on test dataset is: {} ".format(result))

# Initialization of FastAPI
app = FastAPI()

# Open the model and other pickle file
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)


# 1. Get Method
@app.get("/predict")
async def root():
    return {"y": [1, 2, 3]}


# 2. Post Method Base Model
class Item(BaseModel):
    size: int
    nb_rooms: int
    garden: int
    locations: str  # Est, Sud, Nord, Quest


@app.post("/prediction")
async def do_prediction(data: Item):
    pickle_encoder = open("ohe_orientation.pkl", "rb")
    encoder = pickle.load(pickle_encoder)
    arr = encoder.transform([[data.locations]])
    rem_arr = [[data.size, data.nb_rooms, data.garden]]
    user_input = np.concatenate((rem_arr, arr), axis=1)
    print(user_input.shape)
    prediction = model.predict(user_input)
    print(prediction)
    # return prediction[0]
    return (f"Bonjour, The price of your house is predicted to be Euro {prediction[0]:.2f} , Merci ! ")
