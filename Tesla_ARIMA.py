#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import math


# In[2]:


thr = pd.read_csv("throuput.csv")


# In[63]:


thr.head(50)


# In[4]:


for i in range(len(thr)):
    thr['Interval Ending (AEST)'][i] = datetime.strptime(thr['Interval Ending (AEST)'][i], '%Y/%m/%d %H:%M') 


# In[5]:


split_date = pd.datetime(2018,3,2)
thr1 = thr.loc[thr['Interval Ending (AEST)'] < split_date]


# In[6]:


value = []
for i in range(len(thr1)):
    value.append(0)
thr1['value'] = value


# In[7]:


ratio = []


# In[10]:


for i in range(len(thr1)):
    print(i)
    a = thr1['Raise Throughput Ratio'][i]
    a = float(a)
    b = thr1['Lower Throughput Ratio'][i]
    b = float(b)

    if (math.isnan(a) == True) and (math.isnan(b) == True):   
#         thr1.at[i, 'value'] = 0 
        ratio.append(0)

    elif (math.isnan(a) == True) and (math.isnan(b) == False):
#         thr1.at[i, 'value'] = -b
#         print(-b)
#         print(thr1.at[i, 'value'])
#         print(thr1['value'][i])
        ratio.append(-b)

    elif (math.isnan(a)) == False and (math.isnan(b) == True):
#         thr1.at[i, 'value'] = a
        ratio.append(a)


# In[11]:


thr1['value'] = ratio


# In[12]:


thr1


# In[14]:


x = thr1['Interval Ending (AEST)']


# In[15]:


y = thr1['value']


# In[18]:


plt.figure(figsize=(40,20))
plt.plot(x,y)


# In[19]:


from statsmodels.graphics.tsaplots import plot_acf


# In[20]:


newthr1 = thr1[['Interval Ending (AEST)', 'value']].copy()


# In[25]:


newthr1 = newthr1.set_index('Interval Ending (AEST)')


# In[27]:


plot_acf(newthr1)


# In[29]:


from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error


# In[30]:


len(newthr1)


# In[45]:


# train = newthr1[0:12095]
# test = newthr1[12094:]  
train = newthr1[0:100]
test = newthr1[100:120]
predictions = []


# In[46]:


model_ar = AR(train)
model_ar_fit = model_ar.fit()


# In[48]:


predictions = model_ar_fit.predict(start=100,end=120)


# In[49]:


plt.plot(test)
plt.plot(predictions,color='red')


# In[50]:


predictions


# In[38]:


from statsmodels.tsa.arima_model import ARIMA


# In[76]:


newtrain= newthr1[0:10000]
newtest = newthr1[100:120]


# In[78]:


#p,d,q  p = periods taken for autoregressive model
#d -> Integrated order, difference
# q periods in moving average model
model_arima = ARIMA(newtrain,order=(1, 1, 2))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[75]:


predictions= model_arima_fit.forecast(steps=30)[0]
predictions


# In[58]:


plt.plot(newtest)
#plt.plot(predictions,color='red')


# In[59]:


plt.plot(predictions,color='red')


# In[60]:


mean_squared_error(newtest,predictions)


# In[61]:


import itertools
p=d=q=range(0,5)
pdq = list(itertools.product(p,d,q))
pdq


# In[79]:


import warnings
warnings.filterwarnings('ignore')
minlist = []
for param in pdq:
    try:
        model_arima = ARIMA(newtrain,order=param)
        model_arima_fit = model_arima.fit()
        print(param,model_arima_fit.aic)
        minlist.append(model_arima_fit.aic)
    except:
        continue


# In[80]:


model_arima = ARIMA(newtrain,order=(2, 0, 2))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[81]:


newtest = newthr1[10000:12000]


# In[82]:


predictions= model_arima_fit.forecast(steps=2000)[0]


# In[83]:


mean_squared_error(newtest,predictions)


# In[84]:


plt.plot(newtest)


# In[85]:


plt.plot(predictions,color='red')


# In[86]:


new_train = newthr1[0:12095]
new_test = newthr1[12095:17279]


# In[89]:


model_arima = ARIMA(new_train,order=(2, 0, 2))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[90]:


predictions= model_arima_fit.forecast(steps=5184)[0]


# In[91]:


mean_squared_error(new_test,predictions)


# In[101]:


len(thr)


# In[104]:


value = []
for i in range(len(thr)):
    value.append(0)
thr['value'] = value


# In[112]:


ratio = []


# In[113]:


for i in range(len(thr)):
    print(i)
    a = thr['Raise Throughput Ratio'][i]
    a = float(a)
    b = thr['Lower Throughput Ratio'][i]
    b = float(b)

    if (math.isnan(a) == True) and (math.isnan(b) == True):   
#         thr1.at[i, 'value'] = 0 
        ratio.append(0)

    elif (math.isnan(a) == True) and (math.isnan(b) == False):
#         thr1.at[i, 'value'] = -b
#         print(-b)
#         print(thr1.at[i, 'value'])
#         print(thr1['value'][i])
        ratio.append(-b)

    elif (math.isnan(a)) == False and (math.isnan(b) == True):
#         thr1.at[i, 'value'] = a
        ratio.append(a)
    else:
        ratio.appen(0)


# In[108]:


len(thr)


# In[114]:


len(ratio)


# In[115]:


thr = thr[0:82944]


# In[116]:


len(thr)


# In[117]:


thr['value'] = ratio


# In[118]:


newthr = thr[['Interval Ending (AEST)', 'value']].copy()
newthr = newthr.set_index('Interval Ending (AEST)')


# In[119]:


final_train = newthr[0:60833]
final_test = newthr[60833:82944]


# In[120]:


model_arima = ARIMA(final_train,order=(2, 0, 2))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[121]:


predictions= model_arima_fit.forecast(steps=22111)[0]


# In[122]:


mean_squared_error(final_test,predictions)


# In[123]:


whole_train = newthr


# In[124]:


model_arima = ARIMA(whole_train,order=(2, 0, 2))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[125]:


final_predictions= model_arima_fit.forecast(steps=4032)[0]


# In[127]:


final_predictions


# In[128]:


df = pd.DataFrame(final_predictions)


# In[130]:


df.to_csv("prediction.csv")


# In[ ]:




