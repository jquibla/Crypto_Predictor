# Crypto_Predictor
The objective of this project is to predict if tomorrow Bitcoin will close higher or lower than today.

During the project I use ARIMA and various classification models, several transformers and different combinations of data, including Bitcoin closing price and volume and Gold and Nasdaq Comoposite closing prices.

In order to run the code, please execute the following instructions from your cloned repository to make sure you are using the proper packages:

	- conda env create --file Crypto_Predictor.yml
	
	- conda activate Crypto_Predictor

After that, in order to replicate the project, the notebooks should be run in the order indicated by their names:
	- 01-ARIMA_Model_Bitcoin
	- 02-Bitcoin_Feature_Engineering
	- 03-Gold_Feature_Engineering
	- 04-Nasdaq_Feature_Engineering
	- 05-Model_Selection

If you just want to work with the Frontend (suitable for any user with no data science knowledge), please execute "streamlit run Frontend.py". Please note that the data update proccess uses Firefox for web scraping.
