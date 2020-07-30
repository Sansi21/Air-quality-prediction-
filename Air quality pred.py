#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("multivariate_Train.csv")


# In[ ]:


df.head()


# In[ ]:


df.values.shape


# In[3]:


y_train = df['target'].values


# In[ ]:


y_train.shape


# In[4]:


x_train = df.values[:,0:5]


# In[ ]:


x_train.shape


# In[5]:


m = x_train.shape[0]
one = np.ones((m,1))
x_train = np.hstack((one,x_train))


# In[ ]:


t = x_train.mean()
s = x_train.std()


# In[ ]:


x_train = (x_train - t)/s


# In[ ]:


def hypothesis(X,theta):
    return np.dot(X,theta)


# In[ ]:


def gradient(X,Y,theta):
    m = X.shape[0]
    hx = hypothesis(X,theta)
    grad = np.dot(X.T , hx - Y)
    
    return grad/m


# In[ ]:


def gradient_descent(X,Y,lr = 0.1, max_itr = 100):
    n = X.shape[1]
    theta = np.zeros((n,))
    for i in range(max_itr):
        grad = gradient(X,Y,theta)
        theta = theta - lr*grad
    return theta


# In[ ]:


theta = gradient_descent(x_train,y_train)


# In[ ]:


theta


# In[7]:


dfx = pd.read_csv("multivariate_Test.csv")


# In[8]:


x_test = dfx.values[:,:5]


# In[9]:


m = x_test.shape[0]
one = np.ones((m,1))


# In[10]:


x_test = np.hstack((one,x_test))


# In[ ]:


x_test.shape


# In[ ]:


y_test = hypothesis(x_test,theta)


# In[ ]:


y_test.shape


# In[17]:


dfy = pd.DataFrame(data = a,columns = ['Id'])


# In[ ]:


dfy['target'] = y_test


# In[16]:


a = np.arange(0,400)


# In[ ]:


dfy.to_csv("air_pred",index=False)


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


lr = LinearRegression()


# In[13]:


lr.fit(x_train,y_train)


# In[14]:


y_ = lr.predict(x_test)


# In[15]:


y_.shape


# In[18]:


dfy['target'] = y_


# In[19]:


dfy.to_csv("air_pred1",index=False)


# In[ ]:




