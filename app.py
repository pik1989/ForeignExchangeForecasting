   
from flask import Flask,render_template,redirect,request
from flask.helpers import flash
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fbprophet import Prophet
import warnings

from werkzeug import debug
warnings.filterwarnings('ignore')
from random import randint
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
from flask_socketio import SocketIO
import datetime
from datetime import timedelta, date
from fbprophet.plot import plot_plotly

from tempfile import TemporaryDirectory





def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
    

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

#inr_df.pop('Rate')
#inr_df.pop('Change')
inr_df.head(5)
inr_df = pd.concat([hist_df, inr_df], ignore_index=True)

length=len(inr_df)
data_day1=inr_df[length-1:]
data_day2=inr_df[length-2:length-1]
data_day7=inr_df[length-7:length-6]
data_day15=inr_df[length-15:length-14]
data_day365=inr_df[length-365:length-364]

change_1=float(data_day2['Units per HKD'])-float(data_day1['Units per HKD'])
change_7=float(data_day7['Units per HKD'])-float(data_day1['Units per HKD'])
change_15=float(data_day15['Units per HKD'])-float(data_day1['Units per HKD'])
change_365=float(data_day365['Units per HKD'])-float(data_day1['Units per HKD'])

price_day1=float(data_day1['Units per HKD'])
#print(price_day1,change_1,change_7,change_15,change_365)




import os
app = Flask("__name__")
app.config["IMAGE_UPLOADS"] = "static/img/"
app.config["Graph_UPLOADS"] = "static/graph/"
app.secret_key = "dadbn2e298ynce8y998c@_shlbsjda"
socketio = SocketIO(app)
@app.route('/')
def index():

    
    

    actual_chart = go.Scatter(y=inr_df["Units per HKD"], x=inr_df["Date"], name= 'Data')
    

    with TemporaryDirectory() as tmp_dir:
        filename = tmp_dir + "tmp.html"
        py.plot([actual_chart],filename = filename , auto_open=False)
        with open(filename, "r") as f:
            graph_html = f.read()

    
    IS_FORECAST = False
    return render_template("step1.html",price_day1=price_day1,change_1=change_1,change_7=change_7,change_15=change_15,change_365=change_365, graph_html=graph_html, IS_FORECAST=IS_FORECAST)







@app.route('/submit',methods=['POST'])
def submit_data():
    try:
        s2=int(request.form['parameter'])
        s1=request.form['options']
    except:
        flash("Please provide valid inputs")
        return redirect("/")

    df= inr_df.drop(['Currency', 'Name', 'HKD per unit'], axis=1)
    df = df.rename(columns={'Units per HKD': 'y', 'Date': 'ds'})

    # to save a copy of the original data..you'll see why shortly. 
    df['y_orig'] = df['y'] 
    # log-transform of y
    df['y'] = np.log(df['y'])
    #instantiate Prophet
    model = Prophet() 
    model.fit(df)
    future_data = model.make_future_dataframe(periods=s2, freq = s1)  #dropdown   
    future_data.tail()
    forecast_data = model.predict(future_data)
    forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)

    # make sure we save the original forecast data
    forecast_data_orig = forecast_data 
    forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
    df['y_log']=df['y'] 
    df['y']=df['y_orig']
    final_df = pd.DataFrame(forecast_data_orig)

    actual_chart = go.Scatter(y=df["y_orig"], name= 'Actual')
    predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
    predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
    predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')

    

    with TemporaryDirectory() as tmp_dir:
        filename = tmp_dir + "tmp.html"
        py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower],filename = filename, auto_open=False)
        with open(filename, "r") as f:
            graph_html = f.read()
    if s1=="D":
        value="Days"
    elif s1=="M":
        value="Month"
    else:
        value="Year"
    final_df_1=final_df[['ds','yhat']].tail(s2)
    final_df_1 = final_df_1.rename(columns={'yhat': 'Units Per HKD', 'ds':value})
    final_df_1.reset_index(drop=True, inplace=True)
    IS_FORECAST = True
    
    table = final_df_1.to_html(classes='table table-striped', border=0)
    table = table.replace('tr style="text-align: right;"', 'tr style="text-align: center;"')
    table = table.replace('<th></th>', '')
    table = table.replace('<th>', '<th colspan="2">', 1)
    print(table)
    return render_template("step1.html",price_day1=price_day1,change_1=change_1,change_7=change_7,change_15=change_15,change_365=change_365, graph_html=graph_html, parameter=s2,table=table, IS_FORECAST = IS_FORECAST)


   

    
if __name__ =="__main__":

    socketio.run(app, port=8000, debug=True)
