#!/usr/bin/env python
# coding: utf-8

# # # BHARAT INTERN
# # 
# # # NAME-SIDAGAM SURYA DEEPAK
# # 
# # # TASK1-STOCK PREDICTION
# #  - IN THIS WE WILL USE THE KSU-STOCK-PREDICTION-WITH-lSTM DATASET FOR STOCK PREDICTION
# import pandas as pd
# import numpy as np
# import tensorflow.compat.v1 as tf
# import keras
# import matplotlib.pyplot as plt
# import math
# import time
# import warnings
# warnings.filterwarnings('ignore')
# plt.style.use('fivethirtyeight')
# from sklearn import metrics
# import numpy as np
# import pandas as pd
# import math
# import sklearn
# import sklearn.preprocessing
# import datetime
# import os
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
# 
# 
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
# from sklearn.metrics import mean_squared_error, r2_score
# from plotly import __version__
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import cufflinks as cf
# import plotly.offline as pyo
# cf.go_offline()
# pyo.init_notebook_mode()
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# print(__version__)
# import os
# for dirname, _, filenames in os.walk('kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Loading our data

# In[2]:


df = pd.read_csv("../input/nyse/prices-split-adjusted.csv")
plot_x = df['date'].copy()
df.set_index("date", inplace = True)
df.index = pd.to_datetime(df.index)
df.head()


# # Checking for duplicates and null values

# In[3]:


#Check for duplicated values
df.duplicated().sum()


# In[4]:


df.drop_duplicates(inplace=True)


# In[5]:


#Check for null values
df.isna().sum()


# In[6]:


symbols = list(set(df.symbol))
len(symbols)


# # Preprocessing

# In[7]:


plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);
plt.plot(df[df.symbol == 'EQIX'].open.values, color='red', label='open')
plt.plot(df[df.symbol == 'EQIX'].close.values, color='green', label='close')
plt.plot(df[df.symbol == 'EQIX'].low.values, color='blue', label='low')
plt.plot(df[df.symbol == 'EQIX'].high.values, color='black', label='high')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
#plt.show()

plt.subplot(1,2,2);
plt.plot(df[df.symbol == 'EQIX'].volume.values, color='black', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best');


# In[8]:


df.symbol.value_counts()


# We will choose one specific stock to build our model so will choose **KSU** stock

# In[9]:


# Scalling
KSU_stock = df[df['symbol'] == 'KSU']

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
KSU_df = KSU_stock.copy()
KSU_df.drop(['symbol'], axis=1, inplace=True)
x = KSU_df[['open', 'low', 'high', 'volume']].copy()
y = KSU_df['close'].copy()

x[['open', 'low', 'high', 'volume']] = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y.values.reshape(-1, 1))



# In[10]:


#Splitting

def load_data(X, seq_len, train_size=0.9):
    amount_of_features = X.shape[1]
    X_mat = X.values
    sequence_length = seq_len + 1
    data = []
    
    for index in range(len(X_mat) - sequence_length):
        data.append(X_mat[index: index + sequence_length])
    
    data = np.array(data)
    train_split = int(round(train_size * data.shape[0]))
    train_data = data[:train_split, :]
    
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1][:,-1]
    
    x_test = data[train_split:, :-1] 
    y_test = data[train_split:, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return x_train, y_train, x_test, y_test

window = 22
x['close'] = y
X_train, y_train, X_test, y_test = load_data(x, window)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# ### Visualize our KSU stock

# In[11]:


plt.figure(figsize=(15, 5));
plt.plot(KSU_stock.open.values, color='red', label='open')
plt.plot(KSU_stock.close.values, color='green', label='low')
plt.plot(KSU_stock.low.values, color='blue', label='low')
plt.plot(KSU_stock.high.values, color='black', label='high')
#plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
plt.title('stock')
plt.xlabel('time [days]')
plt.ylabel('price/volume')
plt.legend(loc='best')
plt.show()


# # Modeling

# # LSTM architecture
# model = Sequential()
# # First LSTM layer with Dropout regularisation
# model.add(LSTM(units=50, input_shape=(window,5),return_sequences=True))
# model.add(Dropout(0.2))
# # Second LSTM layer
# model.add(LSTM(units=50,return_sequences=True))
# model.add(Dropout(0.2))
# # Third LSTM layer
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# # Fourth LSTM layer
# model.add(LSTM(units=50))
# model.add(Dropout(0.5))
# # The output layer
# model.add(Dense(units=50, kernel_initializer='uniform', activation='tanh'))
# model.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))
# 
# earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
# callbacks_list = [earlystop]
# 
# # Compiling the RNN
# model.compile(optimizer='adam',loss='mean_squared_error')
# # Fitting to the training set
# start = time.time()
# LSTM=model.fit(X_train,y_train,epochs=100,batch_size=35, validation_split=0.05, verbose=1,callbacks=callbacks_list)
# print ('compilation time : ', time.time() - start)

#  

# In[13]:


model.summary()


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
losses = pd.DataFrame(LSTM.history)
losses.plot()


# # Prediction

# In[15]:


trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

trainPredict = y_scaler.inverse_transform(trainPredict)
trainY = y_scaler.inverse_transform([y_train])
testPredict = y_scaler.inverse_transform(testPredict)
testY = y_scaler.inverse_transform([y_test])


# In[16]:


plot_predicted = testPredict.copy()
plot_predicted = plot_predicted.reshape(174, 1)
plot_actual = testY.copy()
plot_actual = plot_actual.reshape(174, 1)
print(plot_actual.shape)
print(plot_predicted.shape)


# In[17]:


plt.figure(figsize=(20,7))
plot_x = pd.to_datetime(plot_x.iloc[-174:])
plt.plot(pd.DataFrame(plot_predicted), label='Predicted')
plt.plot(pd.DataFrame(plot_actual), label='Actual')
plt.legend(loc='best')
plt.show()


# In[18]:


trainScore = metrics.mean_squared_error(trainY[0], trainPredict[:,0]) ** .5
print('Train Score: %.2f RMSE' % (trainScore))
testScore = metrics.mean_squared_error(testY[0], testPredict[:,0]) ** .5
print('Test Score: %.2f RMSE' % (testScore))


# In[19]:


KSU_stock_prices = KSU_stock.close.values.astype('float32')
KSU_stock_prices = KSU_stock_prices.reshape(len(KSU_stock_prices), 1)


# In[20]:


trainPredictPlot = np.empty_like(KSU_stock_prices)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[22:len(trainPredict)+22, :] = trainPredict

testPredictPlot = np.empty_like(KSU_stock_prices)
testPredictPlot[:, :] = np.nan
testPredictPlot[(len(KSU_stock_prices) - testPredict.shape[0]):len(KSU_stock_prices), :] = testPredict


# In[21]:


plt.figure(figsize=(20,7))
plt.plot(pd.DataFrame(KSU_stock_prices, columns=["close"], index=KSU_df.index).close, label='Actual')
plt.plot(pd.DataFrame(trainPredictPlot, columns=["close"], index=KSU_df.index).close, label='Training')
plt.plot(pd.DataFrame(testPredictPlot, columns=["close"], index=KSU_df.index).close, label='Testing')
plt.legend(loc='best')
plt.show()


# In[22]:


model.save('./Final_model.h5')


# # LSTM architecture visualization

# In[23]:


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
from IPython.display import Image
Image("model.png")


# In[24]:


get_ipython().system('pip install keras_sequential_ascii')
from keras_sequential_ascii import keras2ascii
keras2ascii(model)


# In[ ]:





# In[ ]:




