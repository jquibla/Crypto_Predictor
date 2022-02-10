import numpy as np
from PIL import Image
import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from Model_Frontend import data_engineering, run_model, predict_for_date, harvest_data

# Reading Bitcoin closing price for each date
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

title_image = Image.open("./img/bitcoin_logo.png")
st.image(title_image)

st.markdown("You can use this app to ***predict if Bitcoin price (in dollars) will go up or down tomorrow***.")
st.markdown("This app uses machine learning based on a technical analysis approach for its predictions.")
st.markdown("For simulations, it assumes that you buy or sell at exactly the closing price for the day (UTC time).")


sep_image = Image.open("./img/orange-divider-line-300x76.jpg")
st.image(sep_image)


st.markdown("# **Data**")

data_image = Image.open("./img/data.jpeg")
st.image(data_image)


col1, col2 = st.columns([5,5])
with col1:
	st.markdown("#### Bitcoin price data is available up to " + str(last_date.strftime('%Y-%m-%d')) + ", in this app right now.")
with col2:
	st.markdown("")
	# This function is in Model_Frontend.py
	# It updates the file with the price data using web scraping
	order_update = st.button("Update Bitcoin data", on_click=harvest_data, args=[btc_price])


st.image(sep_image)



st.markdown("# **Predictions**")
pred_image = Image.open("./img/predictions.jpeg")
st.image(pred_image)

col1, col2 = st.columns([5,5])
with col1:
	st.markdown("The box below shows the " + str((last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')) \
		+ " closing price used for the prediction for " \
		+ str((last_date + datetime.timedelta(days=2)).strftime('%Y-%m-%d')) + ". Right now that closing price is unknown, " \
		+ "so change it if you want to see the prediction with a different price or use the \"Update Bitcoin data\" above.")
	closing_price_today = st.number_input('Closing price ' + str((last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')), min_value=0.001, value=btc_price['Close'].values[-1])
with col2:
	# This function is in Model_Frontend.py
	# It just predicts if Bitcoin price will increase or drop in a given date
	st.markdown("## **Prediction for " + str((last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')) + ": " + predict_for_date(btc_price, last_date+datetime.timedelta(days=1)) + "**")
	# In this call I include the input closing price because it's not in the file with prices
	st.markdown("## **Prediction for " + str((last_date + datetime.timedelta(days=2)).strftime('%Y-%m-%d')) + ": " \
		+ predict_for_date(btc_price, last_date+datetime.timedelta(days=2), closing_price_today) + "**")



st.image(sep_image)


st.markdown("# **Simulation**")

sim_image = Image.open("./img/simulation.jpeg")
st.image(sim_image)


st.markdown("###### **You don't trust the model without trying it? That's normal.**")

col1, col2 = st.columns([5,5])
with col1:
	st.markdown("### **Simulate using the model between two dates (will take several seconds)**")
	st.markdown("###### When the simulation is run, the buying points (green) and selling points (red) will be shown")
with col2:
	st.markdown("")
	st.markdown("")
	# The simulation is only called when you press the button. That way any change in period or fee doesn't cause the simulation to be executed
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
	
	if simulate:
		# This function is in Model_Frontend.py
		# It creates all the features needed for the model using the daily price dataframe
		data_for_model = data_engineering(btc_price)
		# This function is in Model_Frontend.py
		# It runs the model in a walk-forward fashion between 2 given dates and using a given transaction fee
		# This function returns the results for the naive model and the chosen model
		# and also the dates when the model indicated bitcoin should be bought and sold
		result_naive, result_model, buying_points, selling_points = run_model(data_for_model, d_range[0], d_range[1], fee/100)

		# Here we add the price to the dates for buying and selling, so that the markers appear at the correct height on the plot
		buying_points = data_show.loc[buying_points.index]
		selling_points = data_show.loc[selling_points.index]
		plt.plot(buying_points, '^', color='green', markersize=7);
		plt.plot(selling_points, 'v', color='red', markersize=7);
	
	buf = BytesIO()
	fig.savefig(buf, format="png")
	st.image(buf)
with col2:
	if simulate:
		st.markdown("")
		st.markdown("")
		st.markdown("### **Bitcoin variation in period: " + str(result_naive) + "**")
		st.markdown("### **Investment variation using model: " + str(result_model) + "**")
		st.bar_chart(data=pd.DataFrame([result_naive, result_model], index=['Bitcoin', 'Model'], columns=['Performance']))
	else:
		st.markdown("")
		st.markdown("")
		st.markdown("### **Press 'Launch simulation' to view the results**")



st.image(sep_image)

st.markdown("Github repo: [Crypto_Predictor](https://github.com/jquibla/Crypto_Predictor)")
