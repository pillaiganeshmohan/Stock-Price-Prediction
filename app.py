from itertools import chain
from pickletools import optimize
from turtle import title
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import datetime
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from joblib import dump, load
from tensorflow import keras
from tensorflow.keras import layers
import os
from flask import Flask, render_template, url_for

app = Flask(__name__, template_folder='Templates', static_folder='Static')

@app.route('/')
def home():

   

    return render_template('home.html')

current_dir = os.path.dirname(os.path.abspath(__file__))   

@app.route('/tcs')
def tcs():
    csv_path = os.path.join(current_dir, "tcs_stock.csv")
    stock1 = pd.read_csv(csv_path)
    print(csv_path)
    #fig1 = px.line(stock1, x="Date", y="Prev Close")
    #fig2 = px.line(stock1,y = stock1['Close'])
    stk1 = stock1.reset_index()['Close']
    stock1["Date"] = stock1["Date"].str.replace('/','-').astype(object)
    scaler=MinMaxScaler(feature_range=(0,1))
    stk1=scaler.fit_transform(np.array(stk1).reshape(-1,1))
    train_size = int(len(stk1)*0.65)
    test_size = len(stk1)-train_size
    train_data,test_data=stk1[0:train_size,:],stk1[train_size:len(stk1),:1]
    train_size,test_size
    def create_data(data,time_step=1):
        dataX, dataY = [], []
        for i in range(len(data)-time_step-1):
            a = data[i:(i+time_step),0]
            dataX.append(a)
            dataY.append(data[i+time_step,0])
        return np.array(dataX), np.array(dataY)
    time_step = 100
    X_train, Y_train = create_data(train_data,time_step)
    X_test, Y_test = create_data(train_data,time_step)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)  
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    model=Sequential()
    model.add(LSTM(50,return_sequences=True , input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    look_back=63
    trainPredictPlot = np.empty_like(stk1)
    trainPredictPlot[:, :]= np.nan
    trainPredictPlot [look_back:len (train_predict)+look_back, :] = train_predict
    testPredictPlot = np.empty_like(stk1) 
    testPredictPlot[:, :]= np.nan
    testPredictPlot[len (train_predict)+(look_back*2)+1: len(stk1)-1, :] = test_predict
    x_input=test_data[37:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
   
    lst_output=[]
    n_steps=50
    i=0
    while(i<30):
        if(len(temp_input)>50):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input=x_input.reshape((1,n_steps,1))
            yhat = model.predict(x_input,verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape(1,n_steps,1)
            yhat = model.predict(x_input,verbose=0 )
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    
    day_new=np.arange(1,51)
    day_pred=np.arange(51,81)
    stk3=stk1.tolist()
    stk3.extend(lst_output)

    y_pred = scaler.inverse_transform(lst_output)
    y_pred = y_pred.tolist()    
    y_pred = list(chain.from_iterable(y_pred))
    y_new= scaler.inverse_transform(stk1[198:])
    y_new=y_new.tolist()
    y_new = list(chain.from_iterable(y_new))
    day_pred=np.arange(51,81)
    day_new=np.arange(1,51)
    day_new = day_new.tolist()
    day_pred = day_pred.tolist()
    fig = px.line(x=day_pred, y=y_pred)
    fig.add_trace(go.Scatter(x = day_new, y = y_new, mode = 'lines'))
    fig.update_xaxes(title="No. of Days")
    fig.update_yaxes(title="Stock price")
    fig.write_image("Static\\actual_vs_pred2.svg",width = 500, height = 300)


    return render_template('tcs.html',title = "tcs")

@app.route('/infosys')
def infosys():
    csv_path = os.path.join(current_dir, "infy_stock.csv")
    stock1 = pd.read_csv(csv_path)
    #fig1 = px.line(stock1, x="Date", y="Prev Close")
    #fig2 = px.line(stock1,y = stock1['Close'])
    stk1 = stock1.reset_index()['Close']
    stock1["Date"] = stock1["Date"].str.replace('/','-').astype(object)
    scaler=MinMaxScaler(feature_range=(0,1))
    stk1=scaler.fit_transform(np.array(stk1).reshape(-1,1))
    train_size = int(len(stk1)*0.65)
    test_size = len(stk1)-train_size
    train_data,test_data=stk1[0:train_size,:],stk1[train_size:len(stk1),:1]
    def create_data(data,time_step=1):
        dataX, dataY = [], []
        for i in range(len(data)-time_step-1):
            a = data[i:(i+time_step),0]
            dataX.append(a)
            dataY.append(data[i+time_step,0])
        return np.array(dataX), np.array(dataY)
    time_step = 100
    X_train, Y_train = create_data(train_data,time_step)
    X_test, Y_test = create_data(train_data,time_step)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)  
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    model=Sequential()
    model.add(LSTM(50,return_sequences=True , input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    look_back=63
    trainPredictPlot = np.empty_like(stk1)
    trainPredictPlot[:, :]= np.nan
    trainPredictPlot [look_back:len (train_predict)+look_back, :] = train_predict
    testPredictPlot = np.empty_like(stk1) 
    testPredictPlot[:, :]= np.nan
    testPredictPlot[len (train_predict)+(look_back*2)+1: len(stk1)-1, :] = test_predict
    x_input=test_data[37:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
   
    lst_output=[]
    n_steps=50
    i=0
    while(i<30):
        if(len(temp_input)>50):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input=x_input.reshape((1,n_steps,1))
            yhat = model.predict(x_input,verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape(1,n_steps,1)
            yhat = model.predict(x_input,verbose=0 )
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    
    day_new=np.arange(1,51)
    day_pred=np.arange(51,81)
    stk3=stk1.tolist()
    stk3.extend(lst_output)

    y_pred = scaler.inverse_transform(lst_output)
    y_pred = y_pred.tolist()    
    y_pred = list(chain.from_iterable(y_pred))
    y_new= scaler.inverse_transform(stk1[198:])
    y_new=y_new.tolist()
    y_new = list(chain.from_iterable(y_new))
    day_pred=np.arange(51,81)
    day_new=np.arange(1,51)
    day_new = day_new.tolist()
    day_pred = day_pred.tolist()
    fig = px.line(x=day_pred, y=y_pred)
    fig.add_trace(go.Scatter(x = day_new, y = y_new, mode = 'lines'))
    fig.update_xaxes(title="No. of Days")
    fig.update_yaxes(title="Stock price")
    fig.write_image("Static\\actual_vs_pred.svg",width = 500, height = 300)

    return render_template('infosys.html',title = "infosys")
    

@app.route('/google')
def google():
    csv_path = os.path.join(current_dir, "google1.csv")
    stock1 = pd.read_csv(csv_path)
    stock1["Date"] = stock1["Date"].str.replace('/','-').astype(object)
    stk1 = stock1.reset_index()['Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    stk1=scaler.fit_transform(np.array(stk1).reshape(-1,1))
    train_size = int(len(stk1)*0.65)
    test_size = len(stk1)-train_size    
    train_data,test_data=stk1[0:train_size,:],stk1[train_size:len(stk1),:1]    
    def create_data(data,time_step=1):
        dataX, dataY = [], []
        for i in range(len(data)-time_step-1):
            a = data[i:(i+time_step),0]
            dataX.append(a)
            dataY.append(data[i+time_step,0])
        return np.array(dataX), np.array(dataY)
    time_step = 100
    X_train, Y_train = create_data(train_data,time_step)
    X_test, Y_test = create_data(train_data,time_step)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    model=Sequential()
    model.add(LSTM(50,return_sequences=True , input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))  
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)    
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    math.sqrt(mean_squared_error(Y_train,train_predict))
    look_back=63
    trainPredictPlot = np.empty_like(stk1)
    trainPredictPlot[:, :]= np.nan
    trainPredictPlot [look_back:len (train_predict)+look_back, :] = train_predict
    testPredictPlot = np.empty_like(stk1) 
    testPredictPlot[:, :]= np.nan
    testPredictPlot[len (train_predict)+(look_back*2)+1: len(stk1)-1, :] = test_predict
    x_input=test_data[37:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=51
    i=0 
    while(i<30):
        if(len(temp_input)>51):
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input)) 
            x_input=x_input.reshape(1,-1)
            x_input=x_input.reshape((1,n_steps,1))
            yhat = model.predict(x_input,verbose=0)
            print("{} day output{}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape(1,n_steps,1)
            yhat = model.predict(x_input,verbose=0 )
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    day_new=np.arange(1,51)
    day_pred=np.arange(51,81)
    stk3=stk1.tolist()
    stk3.extend(lst_output)
    stk3=stk1.tolist()
    stk3.extend(lst_output)

    y_pred = scaler.inverse_transform(lst_output)
    y_pred = y_pred.tolist()    
    y_pred = list(chain.from_iterable(y_pred))
    y_new= scaler.inverse_transform(stk1[198:])
    y_new=y_new.tolist()
    y_new = list(chain.from_iterable(y_new))
    day_pred=np.arange(51,81)
    day_new=np.arange(1,51)
    day_new = day_new.tolist()
    day_pred = day_pred.tolist()
    fig = px.line(x=day_pred, y=y_pred)
    fig.add_trace(go.Scatter(x = day_new, y = y_new, mode = 'lines'))
    fig.update_xaxes(title="No. of Days")
    fig.update_yaxes(title="Stock price")
    fig.write_image("Static\\actual_vs_pred4.svg",width = 500, height = 300)


    return render_template('google.html',title = "google")

@app.route('/nifty')
def nifty():
    csv_path = os.path.join(current_dir, "nifty_it_index.csv")
    stock1 = pd.read_csv(csv_path)
    #fig1 = px.line(stock1, x="Date", y="Prev Close")
    #fig2 = px.line(stock1,y = stock1['Close'])
    stk1 = stock1.reset_index()['Close']
    stock1["Date"] = stock1["Date"].str.replace('/','-').astype(object)
    scaler=MinMaxScaler(feature_range=(0,1))
    stk1=scaler.fit_transform(np.array(stk1).reshape(-1,1))
    train_size = int(len(stk1)*0.65)
    test_size = len(stk1)-train_size
    train_data,test_data=stk1[0:train_size,:],stk1[train_size:len(stk1),:1]
    train_size,test_size
    def create_data(data,time_step=1):
        dataX, dataY = [], []
        for i in range(len(data)-time_step-1):
            a = data[i:(i+time_step),0]
            dataX.append(a)
            dataY.append(data[i+time_step,0])
        return np.array(dataX), np.array(dataY)
    time_step = 100
    X_train, Y_train = create_data(train_data,time_step)
    X_test, Y_test = create_data(train_data,time_step)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)  
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    model=Sequential()
    model.add(LSTM(50,return_sequences=True , input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    look_back=63
    trainPredictPlot = np.empty_like(stk1)
    trainPredictPlot[:, :]= np.nan
    trainPredictPlot [look_back:len (train_predict)+look_back, :] = train_predict
    testPredictPlot = np.empty_like(stk1) 
    testPredictPlot[:, :]= np.nan
    testPredictPlot[len (train_predict)+(look_back*2)+1: len(stk1)-1, :] = test_predict
    x_input=test_data[37:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
   
    lst_output=[]
    n_steps=50
    i=0
    while(i<30):
        if(len(temp_input)>50):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input=x_input.reshape((1,n_steps,1))
            yhat = model.predict(x_input,verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape(1,n_steps,1)
            yhat = model.predict(x_input,verbose=0 )
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    
    day_new=np.arange(1,51)
    day_pred=np.arange(51,81)
    stk3=stk1.tolist()
    stk3.extend(lst_output)

    y_pred = scaler.inverse_transform(lst_output)
    y_pred = y_pred.tolist()    
    y_pred = list(chain.from_iterable(y_pred))
    y_new= scaler.inverse_transform(stk1[198:])
    y_new=y_new.tolist()
    y_new = list(chain.from_iterable(y_new))
    day_pred=np.arange(51,81)
    day_new=np.arange(1,51)
    day_new = day_new.tolist()
    day_pred = day_pred.tolist()
    fig = px.line(x=day_pred, y=y_pred)
    fig.add_trace(go.Scatter(x = day_new, y = y_new, mode = 'lines'))
    fig.update_xaxes(title="No. of Days")
    fig.update_yaxes(title="Stock price")
    fig.write_image("Static\\actual_vs_pred3.svg",width = 500, height = 300)

    return render_template('nifty.html',title = "nifty")

if __name__ == "__main__":
    app.debug = True
    app.run()