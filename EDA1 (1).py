#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data=pd.read_csv("data_clean.csv")
data


# In[4]:


data.info()


# In[5]:


print(type(data))
print(data.shape)


# In[6]:


data.shape


# In[7]:


data.dtypes


# In[ ]:


#drop unecessary coloumns.


# In[11]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[14]:


#convert the month datatype to integer data type
data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[15]:


#checking duplicate rows
data1[data1.duplicated()]


# In[17]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[ ]:




