import numpy as np
from PIL import Image
import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from Model_Frontend import data_engineering, run_model, predict_for_date, harvest_data


btc_price = pd.read_csv('./data_proc/Bitcoin_Price_For_Frontend.csv', index_col='Date')
btc_price.index=pd.to_datetime(arg=btc_price.index,format='%Y-%m-%d')
last_date = btc_price.index[-1]


st.set_page_config(page_title="Crypto Predictor: Bitcoin", 
                   page_icon="chart_with_upwards_trend", 
                   layout='wide')

#--------------------------------- ---------------------------------  ---------------------------------
#--------------------------------- SETTING UP THE APP
#--------------------------------- ---------------------------------  ---------------------------------

st.markdown("# **Crypto Predictor: Bitcoin**")

title_image = Image.open("./img/stonks.jpeg")
st.image(title_image)

st.markdown("You can use this app to ***predict if Bitcoin price (in dollars) will go up or down tomorrow***.")
st.markdown("This app uses machine learning based on a technical analysis approach for its predictions.")
st.markdown("For simulations, it assumes that you buy or sell at exactly the closing price for the day (UTC time).")

col1, col2 = st.columns([5,5])
with col1:
	st.markdown("#### Bitcoin price data is available up to " + str(last_date.strftime('%Y-%m-%d')) + ", in this app right now.")
with col2:
	st.markdown("")
	order_update = st.button("Update Bitcoin data", on_click=harvest_data, args=[btc_price])

st.markdown("")
st.markdown("")
st.markdown("")

col1, col2 = st.columns([5,5])
with col1:
	st.markdown("The box below shows the " + str((last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')) \
		+ " closing price used for the prediction for " \
		+ str((last_date + datetime.timedelta(days=2)).strftime('%Y-%m-%d')) + ". Right now that closing price is unknown, " \
		+ "so change it if you want to see the prediction with a different price")
	closing_price_today = st.number_input('Closing price ' + str((last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')), min_value=0.001, value=btc_price['Close'].values[-1])
with col2:
	st.markdown("## **Prediction for " + str((last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')) + ": " + predict_for_date(btc_price, last_date+datetime.timedelta(days=1)) + "**")

	st.markdown("## **Prediction for " + str((last_date + datetime.timedelta(days=2)).strftime('%Y-%m-%d')) + ": " \
		+ predict_for_date(btc_price, last_date+datetime.timedelta(days=2), closing_price_today) + "**")


#Explain what the code is doing
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("#### **You don't trust the model without trying it? That's normal.**")

col1, col2 = st.columns([5,5])
with col1:
	st.markdown("### **Simulate using the model between two dates (will take several seconds)**")
with col2:
	st.markdown("")
	st.markdown("")
	simulate = st.button("Launch simulation")

col1, col2 = st.columns([5,5])
with col1:
	d_range = st.date_input("date range 2018/01/01 to " + str(last_date.strftime('%Y-%m-%d')), [datetime.date(2021, 10, 1), last_date], \
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
	if simulate:
		data_for_model = data_engineering(btc_price)
		result_naive, result_model = run_model(data_for_model, d_range[0], d_range[1], fee/100)
		st.markdown("")
		st.markdown("")
		st.markdown("### **Bitcoin variation in period: " + str(result_naive) + "**")
		st.markdown("### **Investment variation using model: " + str(result_model) + "**")
		st.bar_chart(data=pd.DataFrame([result_naive, result_model], index=['Bitcoin', 'Model'], columns=['Performance']))
	else:
		st.markdown("")
		st.markdown("")
		st.markdown("### **Press 'Launch simulation' to view the results**")



st.markdown("")
st.markdown("")
st.markdown("Github repo: [Crypto_Predictor](https://github.com/jquibla/Crypto_Predictor)")
