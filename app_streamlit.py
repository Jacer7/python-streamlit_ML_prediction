import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

st.title('House Price Prediction')

st.write('Which kind of house, you would like to buy !')

url = "http://127.0.0.1:8000/prediction"
with st.form("my_form"):
    size = st.number_input(label="Size", step=1)
    nb_rooms = st.number_input(label="no of rooms", step=1)
    garden = st.radio(
        "Do you want garden?",
        (1, 0))

    genre = st.radio(
        "Which part of city?",
        ('Est', 'Nord', 'Ouest', 'Sud'))
    # if genre == 'South':
    #     st.write('You selected South.')
    # else:
    #     st.write("You selected North.")

    submitted = st.form_submit_button("Submit")
    if submitted:

        r = requests.post(url, json={"size": size,
                                     "nb_rooms": nb_rooms,
                                     "garden": garden,
                                     "locations": genre})
        st.write(r.text)

# Read csv

df = pd.read_csv('C:/Users/Jaysor/Desktop/house.csv')
# st.write(df)
# print(df.columns)  # for the backend to view the columns name in the terminal

fig1 = plt.figure()
plt.scatter(df['size'], df['price'])
st.pyplot(fig1)


@st.cache
def load(file='C:/Users/Jaysor/Desktop/house.csv'):
    df = pd.read_csv(file)
    time.sleep(5)
    return df


df = load()
st.write(df)

# print('Session ?')
if 'page' not in st.session_state:
    st.session_state.page = 'accueil'
if st.session_state.page == 'accueil':
    st.write("coucou page acceuil")
# print('avant', st.session_state)


def modif_key_page(v):
    st.session_state.page = v
    print('fn', v, st.session_state)

# Create a side bar
st.sidebar.button('Accueil', on_click=modif_key_page('Home'))
# print('click a', st.session_state)
st.sidebar.button('Page2', on_click=modif_key_page('page2'))
# print('click p', st.session_state)
# https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4
