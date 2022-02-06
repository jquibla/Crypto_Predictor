import numpy as np
from PIL import Image
import streamlit as st
import datetime
#from Helper import load_data, summary_poster


st.set_page_config(page_title="Crypto Predictor: Bitcoin", 
                   page_icon=":money:", 
                   layout='wide')

#--------------------------------- ---------------------------------  ---------------------------------
#--------------------------------- SETTING UP THE APP
#--------------------------------- ---------------------------------  ---------------------------------
title_image = Image.open("./img/stonks.jpeg")
st.image(title_image)

st.markdown("You can use this app to ***predict if Bitcoin price will go up or down tomorrow***")
st.markdown("This app uses machine learning based on a technical analysis approach for its predictions")


#Explain what the code is doing

st.markdown("### **Simulate using the model between two dates:**")

d5 = st.date_input("date range (2018/01/01 to last day with data)", [datetime.date(2018, 1, 1), datetime.date(2021, 9, 25)])
st.write(d5)

fee = st.slider("Transaction fee for simulation (%): ", min_value=0.00,   
                       max_value=1.00, value=0.15, step=0.01)




st.markdown("Github repo: [Crypto_Predictor](https://github.com/jquibla/Crypto_Predictor)")
