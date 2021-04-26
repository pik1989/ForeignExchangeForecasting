# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 23:59:33 2021

@author: pattn
"""

import pandas as pd
import datetime
from datetime import timedelta, date


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
    
#To download historical data    
'''
start_date = date(2011, 1, 1)
end_date = datetime.date.today()

df = pd.DataFrame()
for single_date in daterange(start_date, end_date):
    dfs = pd.read_html(f'https://www.xe.com/currencytables/?from=HKD&date={single_date.strftime("%Y-%m-%d")}')[0]
    dfs['Date'] = single_date.strftime("%Y-%m-%d")
    df = df.append(dfs)
    
    
df.to_csv('hkd_data_2010.csv')
df = pd.read_csv('hkd_data_2010.csv')
inr_df = df[df['Currency'] == 'INR']
inr_df.to_csv('hkd_data_2010_inr.csv')
'''

hist_df = pd.read_csv('hkd_data_2010_inr.csv')   


start_date = date(2021, 4, 23)
end_date = datetime.date.today()


df = pd.DataFrame()
for single_date in daterange(start_date, end_date):
    dfs = pd.read_html(f'https://www.xe.com/currencytables/?from=HKD&date={single_date.strftime("%Y-%m-%d")}')[0]
    dfs['Date'] = single_date.strftime("%Y-%m-%d")
    df = df.append(dfs)
    
df.to_csv('hkd_data.csv')

inr_df = df[df['Currency'] == 'INR']

inr_df.pop('Rate')
inr_df.pop('Change')

inr_df.head(5)

inr_df = pd.concat([hist_df, inr_df], ignore_index=True)

inr_df.plot(x='Date', y='Units per HKD', figsize=(12, 8))



#FB Prophet Part

import pandas as pd
from fbprophet import Prophet 
import plotly.graph_objs as go
import plotly.offline as py
import numpy as np

df= inr_df.drop(['Currency', 'Name', 'HKD per unit'], axis=1)

df = df.rename(columns={'Units per HKD': 'y', 'Date': 'ds'})
#df['ds'] =  pd.to_datetime(df['ds'], format='%d/%m/%Y')
df.head(5)




# to save a copy of the original data..you'll see why shortly. 
df['y_orig'] = df['y'] 
# log-transform of y
df['y'] = np.log(df['y'])

#instantiate Prophet
model = Prophet() 
model.fit(df)




future_data = model.make_future_dataframe(periods=10, freq = 'D')
future_data.tail()


forecast_data = model.predict(future_data)
forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)



# make sure we save the original forecast data
forecast_data_orig = forecast_data 
forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
fig = model.plot(forecast_data_orig)



fig2 = model.plot_components(forecast_data_orig)


df['y_log']=df['y'] 
df['y']=df['y_orig']



# Python
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(model, forecast_data_orig)  # This returns a plotly Figure
py.iplot(fig)



final_df = pd.DataFrame(forecast_data_orig)
actual_chart = go.Scatter(y=df["y_orig"], name= 'Actual')
predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')
py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower])



















