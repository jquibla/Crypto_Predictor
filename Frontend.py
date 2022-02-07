import numpy as np
from PIL import Image
import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from Model_Frontend import data_engineering, run_model


btc_price = pd.read_csv('./data_proc/Bitcoin_Price_For_Frontend.csv', index_col='Date')
btc_price.index=pd.to_datetime(arg=btc_price.index,format='%Y-%m-%d')
last_date = btc_price.index[-1]


st.set_page_config(page_title="Crypto Predictor: Bitcoin", 
                   page_icon="chart_with_upwards_trend", 
                   layout='wide')

#--------------------------------- ---------------------------------  ---------------------------------
#--------------------------------- SETTING UP THE APP
#--------------------------------- ---------------------------------  ---------------------------------

st.title("Crypto Predictor: Bitcoin")

title_image = Image.open("./img/stonks.jpeg")
st.image(title_image)

st.markdown("You can use this app to ***predict if Bitcoin price will go up or down tomorrow***.")
st.markdown("This app uses machine learning based on a technical analysis approach for its predictions.")


st.subheader("Bitcoin price data is available up to " + str(last_date.strftime('%Y-%m-%d')) + ", in this app right now.")

st.markdown("If you want to update the data, press this button:")

order_update = st.button("Update Bitcoin data")


#Explain what the code is doing

st.markdown("### **Simulate using the model between two dates (will take several seconds)**")

col1, col2 = st.columns([5,5])
with col1:
	d_range = st.date_input("date range (2018/01/01 to " + str(last_date.strftime('%Y-%m-%d')) + ")", [datetime.date(2021, 1, 1), last_date], \
		min_value=datetime.date(2018, 1, 1), max_value=last_date)

with col2:
	fee = st.slider("Transaction fee for simulation (%): ", min_value=0.00,   
                       max_value=1.50, value=0.15, step=0.01)

col1, col2 = st.columns([5,5])
with col1:
	data_show=btc_price[d_range[0]:d_range[1]]
	fig, ax = plt.subplots(figsize=(8, 4))
	plt.title("Bitcoin daily closing price")
	ax.plot(data_show)
	buf = BytesIO()
	fig.savefig(buf, format="png")
	st.image(buf)
with col2:
	data_for_model = data_engineering(btc_price)
	result_naive, result_model = run_model(data_for_model, d_range[0], d_range[1], fee/100)
	st.subheader("Bitcoin variation in period: " + str(result_naive))
	st.subheader("Investment variation using model: " + str(result_model))






st.markdown("Github repo: [Crypto_Predictor](https://github.com/jquibla/Crypto_Predictor)")
