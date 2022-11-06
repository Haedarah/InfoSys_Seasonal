import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import datetime
from datetime import date
from datetime import time
from datetime import timedelta
import time
import altair as alt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing


st.set_page_config(
	page_icon=":chart:",
    page_title="INFOSYS Analysis",
    layout="wide",
)



st.header("INFOSYS Stock Analysis")
st.write("The goal of this analysis is to discuss:")
st.write("    1- Whether we can spot patterns -visually- in case we studied the movement of the stock price in each quarter of the year separately.")
st.write("    2- Whether making predictions is more accurate depending on the quarter of the year we are at.")
st.write("    3- Conclusions of the two above points.")

st.write("""---""")

st.header("1- Attempting to find useful patterns in different seasons of the year")
st.write("The idea is to divide each year to its seasons, and check if making predictions depending on that is more accurate or not.")

data=yf.download(tickers="INFY.NS",start="2012-01-01",end=datetime.now())

data=data.reset_index()
season=[]

for i in data.index:
	if data['Date'][i].month==1 or data['Date'][i].month==2 or data['Date'][i].month==3 :
		season.append("WINTER")
	if data['Date'][i].month==4 or data['Date'][i].month==5 or data['Date'][i].month==6 :
		season.append("SPRING")
	if data['Date'][i].month==7 or data['Date'][i].month==8 or data['Date'][i].month==9 :
		season.append("SUMMER")
	if data['Date'][i].month==10 or data['Date'][i].month==11 or data['Date'][i].month==12 :
		season.append("AUTUMN")
data['Season']=season

winter_data=data[data['Season']=='WINTER']
spring_data=data[data['Season']=='SPRING']
summer_data=data[data['Season']=='SUMMER']
autumn_data=data[data['Season']=='AUTUMN']

data=data.reset_index()

Chart1=go.Figure()
Chart1.add_trace(go.Scatter(x=data["index"],y=data["Close"],mode='lines',line=dict(color="grey",width=3),name="Historical Data"))
Chart1.add_trace(go.Scatter(x=data[data['Season']=="WINTER"]["index"],y=data[data['Season']=="WINTER"]["Close"],mode='markers',marker=dict(color="blue",size=4),name="WINTER"))
Chart1.add_trace(go.Scatter(x=data[data['Season']=="SPRING"]["index"],y=data[data['Season']=="SPRING"]["Close"],mode='markers',marker=dict(color="red",size=4),name="SPRING"))
Chart1.add_trace(go.Scatter(x=data[data['Season']=="SUMMER"]["index"],y=data[data['Season']=="SUMMER"]["Close"],mode='markers',marker=dict(color="orange",size=4),name="SUMMER"))
Chart1.add_trace(go.Scatter(x=data[data['Season']=="AUTUMN"]["index"],y=data[data['Season']=="AUTUMN"]["Close"],mode='markers',marker=dict(color="brown",size=4),name="AUTUMN"))
Chart1.update_layout(xaxis_title="Index -representing the last 10 years of the stock performance-", yaxis_title="Close/₹",height=600,
	title={'text':'InfoSys stock seasonal performance','y':0.9,'x':0.5,'xanchor':'center','yanchor': 'top'})
st.plotly_chart(Chart1,use_container_width=True)

st.write("After this general look on the plotting, it doesn't look like we have found anything special regarding any season performance.")
st.write("But to make sure, we will take a deeper look on each season in separate plotting.")
st.write("")


st.subheader("1.1- Winter")
st.write("For each winter in the last 10 years, we will plot the starting value as 0, and the rest of the values will be the change (+/-) of the stock during each day of the season.")
st.write("-This is just to make reading the plot easier, as values of the price might vary a lot from year to year-")

winter_data=winter_data.reset_index(drop=True)

year=[]
modified=[]
virtical=[]
winter_div=0
flag1=0
current_year=2012
current_minus=winter_data['Close'][0]
for i in winter_data.index:
	if current_year==2022 and flag1==0:
		winter_div=i
		flag1=1
	year.append(winter_data['Date'][i].year)
	if winter_data['Date'][i].year!=current_year:
		current_year=current_year+1
		current_minus=winter_data['Close'][i]
		virtical.append(i)
	
	modified.append(winter_data['Close'][i]-current_minus)

winter_data['Year']=year
winter_data['Modified']=modified
winter_data=winter_data.reset_index()

Chart1_1=go.Figure()
Chart1_1.add_trace(go.Scatter(x=winter_data["index"],y=winter_data["Modified"],mode='lines',line=dict(color="blue",width=2),name="WINTER"))
Chart1_1.add_hline(y=0,line_dash="dot")
Chart1_1.add_vline(x=0,line_color="black",line_width=3,annotation_text=str(2012))
for i in range(10):
	Chart1_1.add_vline(x=virtical[i],line_color="black",line_width=3,annotation_text=str(2013+i))
Chart1_1.update_layout(xaxis_title="Index -representing the last 10 winters of the stock performance-", yaxis_title="Change (+/-) of the stock price",height=600,
	title={'text':'InfoSys stock winter performance','y':0.9,'x':0.5,'xanchor':'center','yanchor': 'top'})
st.plotly_chart(Chart1_1,use_container_width=True)


st.subheader("1.2- Spring")
st.write("For each spring in the last 10 years, we will plot the starting value as 0, and the rest of the values will be the change (+/-) of the stock during each day of the season.")
st.write("-This is just to make reading the plot easier, as values of the price might vary a lot from year to year-")

spring_data=spring_data.reset_index(drop=True)

year=[]
modified=[]
virtical=[]
spring_div=0
flag2=0
current_year=2012
current_minus=spring_data['Close'][0]
for i in spring_data.index:
	if current_year==2022 and flag2==0:
		spring_div=i
		flag2=1
	year.append(spring_data['Date'][i].year)
	if spring_data['Date'][i].year!=current_year:
		current_year=current_year+1
		current_minus=spring_data['Close'][i]
		virtical.append(i)
	
	modified.append(spring_data['Close'][i]-current_minus)

spring_data['Year']=year
spring_data['Modified']=modified
spring_data=spring_data.reset_index()

Chart1_2=go.Figure()
Chart1_2.add_trace(go.Scatter(x=spring_data["index"],y=spring_data["Modified"],mode='lines',line=dict(color="red",width=2),name="SPRING"))
Chart1_2.add_hline(y=0,line_dash="dot")
Chart1_2.add_vline(x=0,line_color="black",line_width=3,annotation_text=str(2012))
for i in range(10):
	Chart1_2.add_vline(x=virtical[i],line_color="black",line_width=3,annotation_text=str(2013+i))
Chart1_2.update_layout(xaxis_title="Index -representing the last 10 springs of the stock performance-", yaxis_title="Change (+/-) of the stock price",height=600,
	title={'text':'InfoSys stock spring performance','y':0.9,'x':0.5,'xanchor':'center','yanchor': 'top'})
st.plotly_chart(Chart1_2,use_container_width=True)


st.subheader("1.3- Summer")
st.write("For each summer in the last 10 years, we will plot the starting value as 0, and the rest of the values will be the change (+/-) of the stock during each day of the season.")
st.write("-This is just to make reading the plot easier, as values of the price might vary a lot from year to year-")

summer_data=summer_data.reset_index(drop=True)

year=[]
modified=[]
virtical=[]
summer_div=0
flag3=0
current_year=2012
current_minus=summer_data['Close'][0]
for i in summer_data.index:
	if current_year==2022 and flag3==0:
		summer_div=i
		flag3=1
	year.append(summer_data['Date'][i].year)
	if summer_data['Date'][i].year!=current_year:
		current_year=current_year+1
		current_minus=summer_data['Close'][i]
		virtical.append(i)
	
	modified.append(summer_data['Close'][i]-current_minus)

summer_data['Year']=year
summer_data['Modified']=modified
summer_data=summer_data.reset_index()

Chart1_3=go.Figure()
Chart1_3.add_trace(go.Scatter(x=summer_data["index"],y=summer_data["Modified"],mode='lines',line=dict(color="orange",width=2),name="SUMMER"))
Chart1_3.add_hline(y=0,line_dash="dot")
Chart1_3.add_vline(x=0,line_color="black",line_width=3,annotation_text=str(2012))
for i in range(10):
	Chart1_3.add_vline(x=virtical[i],line_color="black",line_width=3,annotation_text=str(2013+i))
Chart1_3.update_layout(xaxis_title="Index -representing the last 10 summers of the stock performance-", yaxis_title="Change (+/-) of the stock price",height=600,
	title={'text':'InfoSys stock summer performance','y':0.9,'x':0.5,'xanchor':'center','yanchor': 'top'})
st.plotly_chart(Chart1_3,use_container_width=True)


st.subheader("1.4- Autumn")
st.write("For each autumn in the last 10 years, we will plot the starting value as 0, and the rest of the values will be the change (+/-) of the stock during each day of the season.")
st.write("-This is just to make reading the plot easier, as values of the price might vary a lot from year to year-")

autumn_data=autumn_data.reset_index(drop=True)

year=[]
modified=[]
virtical=[]
autumn_div=0
flag4=0
current_year=2012
current_minus=autumn_data['Close'][0]
for i in autumn_data.index:
	if current_year==2022 and flag4==0:
		autumn_div=i
		flag4=1
	year.append(autumn_data['Date'][i].year)
	if autumn_data['Date'][i].year!=current_year:
		current_year=current_year+1
		current_minus=autumn_data['Close'][i]
		virtical.append(i)
	
	modified.append(autumn_data['Close'][i]-current_minus)

autumn_data['Year']=year
autumn_data['Modified']=modified
autumn_data=autumn_data.reset_index()

Chart1_4=go.Figure()
Chart1_4.add_trace(go.Scatter(x=autumn_data["index"],y=autumn_data["Modified"],mode='lines',line=dict(color="brown",width=2),name="AUTUMN"))
Chart1_4.add_hline(y=0,line_dash="dot")
Chart1_4.add_vline(x=0,line_color="black",line_width=3,annotation_text=str(2012))
for i in range(10):
	Chart1_4.add_vline(x=virtical[i],line_color="black",line_width=3,annotation_text=str(2013+i))
Chart1_4.update_layout(xaxis_title="Index -representing the last 10 autumns of the stock performance-", yaxis_title="Change (+/-) of the stock price",height=600,
	title={'text':'InfoSys stock autumn performance','y':0.9,'x':0.5,'xanchor':'center','yanchor': 'top'})
st.plotly_chart(Chart1_4,use_container_width=True)


st.subheader("1.5- Observations")
st.write("As we saw, there is no noticable patterns in the seasonal data.")
st.write("However, in the next section, we will apply an ARIMA Model on each season, to compair the results with the general ARIMA Model, and check if the results will be more accurate.")


st.write("""---""")

st.header("2- Applying ARIMA on each season's data, to check whether we can get better results")
st.write("In this chapter, we will apply an auto ARIMA Model on each season alone.")



st.subheader("2.1- Winter:")

winter_train=winter_data[:winter_div]
winter_test=winter_data[winter_div:]

auto_arima1=auto_arima(winter_train['Modified'],seasonal=False,stepwise=False)
validate=auto_arima1.predict(n_periods=len(winter_test))
winter_data['Validation']=[None]*len(winter_train)+list(validate)

st.write("Auto ARIMA MODEL:"+str(auto_arima1.get_params().get("order")))

Chart2_1=Chart1_1
Chart2_1.add_trace(go.Scatter(x=winter_data["index"],y=winter_data["Validation"],mode='lines',line=dict(color="red",width=3),name="Validation"))
st.plotly_chart(Chart2_1,use_container_width=True)



st.subheader("2.2- Spring:")

spring_train=spring_data[:spring_div]
spring_test=spring_data[spring_div:]

auto_arima2=auto_arima(spring_train['Modified'],seasonal=False,stepwise=False)
validate=auto_arima2.predict(n_periods=len(spring_test))
spring_data['Validation']=[None]*len(spring_train)+list(validate)

st.write("Auto ARIMA MODEL:"+str(auto_arima2.get_params().get("order")))

Chart2_2=Chart1_2
Chart2_2.add_trace(go.Scatter(x=spring_data["index"],y=spring_data["Validation"],mode='lines',line=dict(color="blue",width=3),name="Validation"))
st.plotly_chart(Chart2_2,use_container_width=True)


st.subheader("2.3- Summer:")

summer_train=summer_data[:summer_div]
summer_test=summer_data[summer_div:]

auto_arima3=auto_arima(summer_train['Modified'],seasonal=False,stepwise=False)
validate=auto_arima3.predict(n_periods=len(summer_test))
summer_data['Validation']=[None]*len(summer_train)+list(validate)

st.write("Auto ARIMA MODEL:"+str(auto_arima3.get_params().get("order")))

Chart2_3=Chart1_3
Chart2_3.add_trace(go.Scatter(x=summer_data["index"],y=summer_data["Validation"],mode='lines',line=dict(color="red",width=3),name="Validation"))
st.plotly_chart(Chart2_3,use_container_width=True)



st.subheader("2.4- Autumn:")

autumn_train=autumn_data[:autumn_div]
autumn_test=autumn_data[autumn_div:]

auto_arima4=auto_arima(autumn_train['Modified'],seasonal=False,stepwise=False)
validate=auto_arima4.predict(n_periods=len(autumn_test))
autumn_data['Validation']=[None]*len(autumn_train)+list(validate)

st.write("Auto ARIMA MODEL:"+str(auto_arima4.get_params().get("order")))

Chart2_4=Chart1_4
Chart2_4.add_trace(go.Scatter(x=autumn_data["index"],y=autumn_data["Validation"],mode='lines',line=dict(color="blue",width=3),name="Validation"))
st.plotly_chart(Chart2_4,use_container_width=True)


st.subheader("2.5- Observations")
st.write("WINTER:")
st.write("In the data we are using, the first 8 out of 10 winters of the training dataset are moving slowly and the variance is kind of stable throughout time. The last 2 winters in addition to the winter we are using for validation, are completely different and they aren't following the patterns of the previous winters. So the validation plotting of our MODEL is more like the first 8 winters, with a small variance throughout time.")
st.write("However, the winter's validation is showing bad results, it can't be considered accurate.")
st.write("")
st.write("")
st.write("")
st.write("SPRING:")
st.write("Similar situation is appearing here, the recent spring is the first spring where closing values goes under 100 rupees loss. So the prediction is kind of reasonable depending on the training dataset(The first 10 winters of the dataset).")
st.write("Bad Prediction, even worse than winter's.")
st.write("")
st.write("")
st.write("")
st.write("SUMMER:")
st.write("The model here shows a failure in predicting this summer's values.")
st.write("")
st.write("")
st.write("")
st.write("AUTUMN:")
st.write("Seems good, not super accurate though.")
st.write("Actually it is so far the best season's prediction. However, the testing dataset is still small, since we are still in November of 2022, and as time marches on, errors might occure.")


st.write("""---""")

st.header("3- Conclusions")

st.subheader("  1.There is no sign that InfoSys stock's performance is related to which season of the year we are at, at least with the current parameters I am taking.")
st.subheader("  2.The closing price of a stock, InfoSys for example, is too hard to predict, because it is dependent of many factors that affect the whole financial market (Wars, Political situations, disasters, etc..) and it is less dependent on its historical values.")
st.subheader("  3.One good idea is to find a parameter which the stock depends on, and try to predict the parameter's behaviour, in order to reflect this prediction on the future values of our stock. (I don't know what parameter to choose, I am still looking more into that.)")
st.subheader("  4.As a final attempt to predict the closing price future, I will use LSTM Models in my next report to see if better results will come.")

st.write("---")
