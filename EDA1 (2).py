#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("data_clean.csv")
data


# In[4]:


data.info()


# In[3]:


print(type(data))
print(data.shape)


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[ ]:


#drop unecessary coloumns.


# In[7]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[8]:


#convert the month datatype to integer data type
data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[15]:


#checking duplicate rows
data1[data1.duplicated()]


# In[17]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[13]:


#change the name
data1.rename({'Solar.R':'Solar','Temp':'Temperature'}, axis=1, inplace = True)
data1


# In[14]:


#display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[17]:


#visualize data1 missing values using graph
cols = data1.columns
colours = ['black','yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[22]:


#find the mean and median values
#impitation of missing values with median
median_ozone=data1['Ozone'].median()
mean_ozone=data1['Ozone'].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[23]:


#Replace the ozone missing values with median
data1['Ozone'] = data1["Ozone"].fillna(median_ozone)
data1.isnull().sum()


# In[24]:


#Replace the ozone missing values with mean
data1['Ozone'] = data1["Ozone"].fillna(mean_ozone)
data1.isnull().sum()


# In[ ]:




