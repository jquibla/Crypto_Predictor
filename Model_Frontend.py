import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
import numbers
import datetime
from selenium import webdriver
from time import sleep




# This function performs web scraping and updates the file with Bitcoin prices
def harvest_data(btc_price):

	url = 'https://coinmarketcap.com/currencies/bitcoin/historical-data/'
	driver = webdriver.Firefox()
	driver.get(url)
	#We go to the bottom of the page because the table with prices won't load any other way
	driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
	#We wait a little bit just in case the page takes time to load
	sleep(5)
	url_html = driver.page_source
	#The first table is the one with the prices
	bitcoin_table_sc = pd.read_html(url_html)[0]
	driver.close()

	# We change the format of the scraped table to be the same as the Bitcoin price dataframe
	bitcoin_table_sc = bitcoin_table_sc.set_index('Date')
	bitcoin_table_sc.index = pd.to_datetime(arg=bitcoin_table_sc.index,format='%b %d, %Y')
	bitcoin_table_sc = bitcoin_table_sc.drop(['Open*', 'High', 'Low', 'Volume', 'Market Cap'], axis=1)
	bitcoin_table_sc = bitcoin_table_sc.rename(columns={'Close**': 'Close'})
	bitcoin_table_sc['Close'] = bitcoin_table_sc['Close'].str[1:].str.replace(',','').astype(float)

	last_date_stored = btc_price.index[-1]
	last_date_scraped = bitcoin_table_sc.index[0]

	#We include the new values in the Bitcoin price dataframe
	for date_i in pd.date_range(last_date_stored, last_date_scraped):
		#We modify only the last date included in the Bitcoin price dataframe (just in case it has changed, though it shouldn't)
		if date_i==last_date_stored:
			btc_price.loc[date_i]=bitcoin_table_sc.loc[date_i]
		#We add new rows for dates not previously included
		else:
			btc_price = btc_price.append(bitcoin_table_sc.loc[date_i])
	
	#We store the new and old values into the original file
	btc_price.to_csv('./data_proc/Bitcoin_Price_For_Frontend.csv')



# This function returns the prediction for the specified date.
# last_closing_price is an optional parameter, it's used when the closing price for today is still unknown (ie the day is not over)
# using last_closing_price you can add a price for today in the dataframe btc_price in order to predict using an expected closing price
def predict_for_date(btc_price, date_pred, last_closing_price=0):
	btc_price_m=btc_price.copy()
	if last_closing_price > 0:
		aux_df = pd.DataFrame([last_closing_price], index=[date_pred-datetime.timedelta(days=1)], columns=['Close'])
		aux_df.index.name = 'Date'
		btc_price_m = btc_price_m.append(aux_df)
	
	# I indicate this is not for a simulation, so that the function creates a row for the last date in the dataframe
	data_for_model = data_engineering(btc_price_m, False)

	#Model building
	model=KNeighborsClassifier(n_neighbors=9)
	
	# Training data is all dates before date_pred
	data_train = data_for_model[:date_pred-datetime.timedelta(days=2)].copy()
	# Data for predict is only date_pred
	data_for_pred = data_for_model[date_pred-datetime.timedelta(days=1):date_pred].copy()

	#I separate the variable I want to predict (if bitcoin price will go up or down the following day)
	X_train=data_train.drop(labels=['subida', 'varPSig'], axis=1)
	y_train=data_train['subida']

	X_for_pred = data_for_pred.drop(labels=['subida', 'varPSig'], axis=1)
    
	number_columns = X_train.select_dtypes('number').columns

	# We only use PowerTransformer_yeo-johnson as column transformer
	transf=[
        	('scaler', PowerTransformer(method='yeo-johnson', standardize=False),number_columns)
        	]

	coltr=ColumnTransformer(transformers=transf, remainder='passthrough')
	
	#model training
	X_train_ct = coltr.fit_transform(X_train)
	model.fit(X_train_ct, y_train)
        
	#We predict if the price will go up the next day
	X_for_pred_ct = coltr.transform(X_for_pred)
	y_pred_next = model.predict(X_for_pred_ct)

	if  y_pred_next[0]:
		answer = 'Bitcoin price will increase!'
	else:
		answer = 'Bitcoin price will drop'
	
	return answer



# This function returns a dataframe with the features to be used in classification models
def data_engineering(btc_price, simulation=True):
	btc_hist_m=btc_price.copy()
	btc_hist_m.reset_index(level=0, inplace=True)

    	#Here, we are adding columns with variations from previous day, week and month
	btc_hist_m['C_dia_ant'] = btc_hist_m['Close'].shift(1)
	btc_hist_m['C_dia_ant'].loc[[0]]=btc_hist_m['C_dia_ant'][1]
	btc_hist_m['var_dia_ant']=btc_hist_m['Close']/btc_hist_m['C_dia_ant']-1

	btc_hist_m['C_sem_ant'] = btc_hist_m['Close'].shift(7)

	for i in range(0,7):
		btc_hist_m['C_sem_ant'].loc[[i]]=btc_hist_m['C_sem_ant'][i+7]

	btc_hist_m['var_sem_ant']=btc_hist_m['Close']/btc_hist_m['C_sem_ant']-1

	btc_hist_m['C_mes_ant'] = btc_hist_m['Close'].shift(30)

	for i in range(0,30):
		btc_hist_m['C_mes_ant'].loc[[i]]=btc_hist_m['C_mes_ant'][i+30]

	btc_hist_m['var_mes_ant']=btc_hist_m['Close']/btc_hist_m['C_mes_ant']-1

	#We create the real dataset for classification algorithms. It will contain:
	#day-to-day price variation of the last 30 days
	#week-to-week price variation from 4 weeks before to 12 weeks before
	#month-to-month price variation from 3 months before to 11 months before
	#if the price went up (1) o not (0) the next day. This is what we are trying to predict
	#the actual price variation during the next day
	#all this data starting 2016/1/1

	columnasData=['varP0', 'varP1', 'varP2', 'varP3', 'varP4', 'varP5', 'varP6', 'varP7', 'varP8', 'varP9',
             'varP10', 'varP11', 'varP12', 'varP13', 'varP14', 'varP15', 'varP16', 'varP17', 'varP18', 'varP19',
             'varP20', 'varP21', 'varP22', 'varP23', 'varP24', 'varP25', 'varP26', 'varP27', 'varP28', 'varP29',
             'varPs30', 'varPs37', 'varPs44', 'varPs51', 'varPs58', 'varPs65', 'varPs72', 'varPs79', 'varPs86',
             'varPm93', 'varPm123', 'varPm153', 'varPm183', 'varPm213', 'varPm243', 'varPm273', 'varPm303',
             'varPm333', 'subida', 'varPSig']
	data_for_use = pd.DataFrame(columns=columnasData)


	for i in range(0,btc_hist_m.shape[0]-1):
		if btc_hist_m['Date'][i]>pd.to_datetime(arg='2015-12-31',format='%Y-%m-%d'):
			dtemp = pd.concat([pd.DataFrame(btc_hist_m['var_dia_ant'][i-29:i+1][::-1].values),
                            pd.DataFrame(btc_hist_m['var_sem_ant'][i-92:i-29][::-7].values),
                            pd.DataFrame(btc_hist_m['var_mes_ant'][i-362:i-92][::-30].values),
                            pd.DataFrame([btc_hist_m['var_dia_ant'][i+1]>0]),
                            pd.DataFrame([btc_hist_m['var_dia_ant'][i+1]])],
                          ignore_index=True, axis=0).T

			dtemp = dtemp.set_index(keys=[btc_hist_m['Date'].loc[[i]]])
        
			new_cols = {x: y for x, y in zip(dtemp.columns, data_for_use.columns)}

			dtemp = dtemp.rename(columns=new_cols)

			data_for_use = data_for_use.append(dtemp.rename(columns=new_cols))

	# If I'm not simulating, I need to add the values for the last day, so I can predict the next day.
	# I set column subida to False and column varPSig to 0.01. There's no reason for that,
	# only that I need to have those columns even if I'm not going to use them
	if not simulation:
		i = i+1
		dtemp = pd.concat([pd.DataFrame(btc_hist_m['var_dia_ant'][i-29:i+1][::-1].values),
                            pd.DataFrame(btc_hist_m['var_sem_ant'][i-92:i-29][::-7].values),
                            pd.DataFrame(btc_hist_m['var_mes_ant'][i-362:i-92][::-30].values),
                            pd.DataFrame([False]),
                            pd.DataFrame([0.01])],
                          ignore_index=True, axis=0).T
        
		dtemp = dtemp.set_index(keys=[btc_hist_m['Date'].loc[[i]]])
		new_cols = {x: y for x, y in zip(dtemp.columns, data_for_use.columns)}
		dtemp = dtemp.rename(columns=new_cols)
		data_for_use = data_for_use.append(dtemp.rename(columns=new_cols))

	return data_for_use


# this function runs the model in a walk-forward approach using the period, data and transaction fee provided
def run_model(data_model, start_date, end_date, fee):
	data_for_use_basic = data_model.copy()
	
	model=KNeighborsClassifier(n_neighbors=9)

	# Initial training data includes all data up until the start_date
	data_train=data_for_use_basic[:start_date-datetime.timedelta(days=1)].copy()
	data_cv=data_for_use_basic[start_date:end_date].copy()

	#I separate the variable I want to predict (if bitcoin price will go up or down the following day)
	X_train=data_train.drop(labels=['subida', 'varPSig'], axis=1)
	y_train=data_train['subida']
	real_train=data_train['varPSig']

	X_test=data_cv.drop(labels=['subida', 'varPSig'], axis=1)
	y_test=data_cv['subida']
	real_test=data_cv['varPSig']
    
	number_columns = X_train.select_dtypes('number').columns

	# as explained above, we only use PowerTransformer_yeo-johnson as column transformer
	transf=[
        	('scaler', PowerTransformer(method='yeo-johnson', standardize=False),number_columns)
        	]

	coltr=ColumnTransformer(transformers=transf, remainder='passthrough')

    
	conf_mat, f1, acc, res_naive, res_mod, res_comis, buying_points, selling_points = \
            walk_forward_validation (model, X_train, y_train, X_test, y_test, real_test, coltr, fee)

	return res_naive, res_comis, buying_points, selling_points




# My walk forward validation function receives the following parameters:
# model: the actual model that is going to be used to make predictions
# X_train_wfv: training data, without the target variable
# y_train_wfv: target variable for training data (True if the price went up the following day)
# X_test_wfv: test data, without the target variable
# y_test_wfv: target variable for test data (True if the price went up the following day)
# real_test_wfv: the actual price variation for the following day for test data (used to calculate returns)
# ct_wfv: the transformer that it's going to be used on the data
# fee: transaction fee applied
def walk_forward_validation (model, X_train_wfv, y_train_wfv, X_test_wfv, y_test_wfv, real_test_wfv, ct_wfv, fee=0.0015):
	y_pred_wfv = list()

	#We will store the dates where the model buys and sells Bitcoin
	buying_points = pd.DataFrame(columns=['Close'])
	selling_points = pd.DataFrame(columns=['Close'])

    	#variable that indicates if we are invested in bitcoin or not following the model predictions
	dentro=True
	resultado_naive=1
	resultado_mod=1
	resultado_mod_comis=1
    
    
    
	for i in range(len(y_test_wfv)):
		X_train_wfv_ct=ct_wfv.fit_transform(X_train_wfv)
		model.fit(X_train_wfv_ct, y_train_wfv)
		X_test_wfv_ct=ct_wfv.transform(X_test_wfv)
        
		#We predict if the price will go up the next day and we add it to the predictions list
		y_pred_next = model.predict(X_test_wfv_ct[i:i+1])
		y_pred_wfv.append(y_pred_next[0])
        
		#We append the test data for the following day to the training data,
		#so it can be used to train the model in the next iteration of the loop
		X_train_wfv=X_train_wfv.append(X_test_wfv[i:i+1])
		y_train_wfv=y_train_wfv.append(pd.Series(y_test_wfv[i]))
        
        
		#If we change our invested situation (buy or sell), we apply the transaction fee
		if y_pred_next[0]!=dentro:
			resultado_mod_comis=resultado_mod_comis*(1-fee)
			# Storing dates when the model buys or sells Bitcoin
			if y_pred_next[0]:
				if buying_points.size==0:
					buying_points = pd.DataFrame([0], index=[X_test_wfv.index[i-1]], columns=['Close'])
				else:
					buying_points = buying_points.append(pd.DataFrame([0], index=[X_test_wfv.index[i-1]], columns=['Close']))
			else:
				if selling_points.size==0:
					selling_points = pd.DataFrame([0], index=[X_test_wfv.index[i-1]], columns=['Close'])
				else:
					selling_points = selling_points.append(pd.DataFrame([0], index=[X_test_wfv.index[i-1]], columns=['Close']))

        	#We set the invested situation according to the model's prediction
		dentro=y_pred_next[0]
        
        	#Naive model is always invested
		resultado_naive=resultado_naive*(1+real_test_wfv[i])
        	#If we are invested, we apply the price variation for the following day
		if dentro:
			resultado_mod=resultado_mod*(1+real_test_wfv[i])
			resultado_mod_comis=resultado_mod_comis*(1+real_test_wfv[i])
    
    	#The function returns the confusion matrix, f1_score, accuracy and the returns for the naive model
    	#and the model used, without applying the fees and applying them,
	#and the dates when the model used bought and sold bitcoins
	return metrics.confusion_matrix(y_test_wfv, y_pred_wfv), metrics.f1_score(y_test_wfv,y_pred_wfv), \
		metrics.accuracy_score(y_test_wfv,y_pred_wfv), resultado_naive, resultado_mod, resultado_mod_comis, \
		buying_points, selling_points

	


