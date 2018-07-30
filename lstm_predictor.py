import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



parser = argparse.ArgumentParser()
parser.add_argument("--input", help="the input dataset, representing a 1 variable timeseries")
parser.add_argument("--coltime", help="the column representing the time")
args = parser.parse_args()

series = pd.read_csv(args.input)

series.set_index(args.coltim,inplace = True)



pyplot.figure(figsize=(20,6))
pyplot.plot(series.values)
pyplot.show()


pyplot.figure(figsize=(20,6))
pyplot.plot(series.values[:50])
pyplot.show()


scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(series.values)
series = pd.DataFrame(scaled)


window_size = 24

series_s = series.copy()
for i in range(window_size):
    series = pd.concat([series, series_s.shift(-(i+1))], axis = 1)
    
series.dropna(axis=0, inplace=True)
series.head()

series.shape

nrow = round(0.8*series.shape[0])

train = series.iloc[:nrow, :]
test = series.iloc[nrow:,:]

from sklearn.utils import shuffle
#train = shuffle(train)

train_X = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

train_X = train_X.values
train_y = train_y.values
test_X = test_X.values
test_y = test_y.values


# In[16]:


print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)


# In[17]:


train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)


# In[18]:


print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)


# In[28]:


# Define the LSTM model
model = None
model = Sequential()
model.add(LSTM(input_shape = (24,1), output_dim= 24, return_sequences = True))
#model.add(Dropout(0.5))
model.add(LSTM(512))
#model.add(LSTM(256))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="adam", metrics=['mse', 'mae', 'mape', 'cosine'])
model.summary()


# In[ ]:


start = time.time()
model.fit(train_X,train_y,batch_size=512,epochs=50,validation_split=0.15)
print("> Compilation Time : ", time.time() - start)


# In[23]:


# Doing a prediction on all the test data at once
preds = model.predict(test_X)


# In[24]:


preds = scaler.inverse_transform(preds)


# In[25]:


test_y.reshape(-1, 1) 
actuals = scaler.inverse_transform(test_y.reshape(-1,1))
#actuals = test_y


# In[26]:


print(mean_squared_error(actuals,preds))
np.sqrt(mean_squared_error(actuals,preds))


# In[27]:


pyplot.plot(actuals)
pyplot.plot(preds)
pyplot.show()

#Forecasting step by step on the test data set, 


# In[ ]:


pyplot.plot(actuals[:550])
pyplot.plot(preds[:550])
pyplot.show()


# In[ ]:


pyplot.plot(preds)
pyplot.show()


# In[ ]:


preds.shape

### IF we want to be able to predict more than 1 step, so t+2, t+3 and so on,
### we can use a sliding window approach and execute the following code

def moving_test_window_preds(n_future_preds):

    ''' n_future_preds - Represents the number of future predictions we want to make
                         This coincides with the number of windows that we will move forward
                         on the test data
    '''
    preds_moving = []                                    # Use this to store the prediction made on each test window
    moving_test_window = [test_X[0,:].tolist()]          # Creating the first test window
    moving_test_window = np.array(moving_test_window)   # Making it an numpy array
    
    for i in range(n_future_preds):
        preds_one_step = model.predict(moving_test_window) # Note that this is already a scaled prediction so no need to rescale this
        preds_moving.append(preds_one_step[0,0]) # get the value from the numpy 2D array and append to predictions
        preds_one_step = preds_one_step.reshape(1,1,1) # Reshaping the prediction to 3D array for concatenation with moving test window
        moving_test_window = np.concatenate((moving_test_window[:,1:,:], preds_one_step), axis=1) # This is the new moving test window, where the first element from the window has been removed and the prediction  has been appended to the end
    
    preds_arr = np.array(preds_moving).reshape(-1, 1)
    preds_moving = scaler.inverse_transform(preds_arr)
    
    return preds_moving
        


# In[ ]:


preds_moving = moving_test_window_preds(24)


# In[ ]:


pyplot.plot(actuals)
pyplot.plot(preds_moving)
pyplot.show()
#Feed the previous prediction back into the input window by moving it one step forward and then predict at the current time step.


# In[ ]:


pyplot.plot(actuals[:200])
pyplot.plot(preds_moving[:200])
pyplot.show()

