#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


data=pd.read_csv("Newspaperdata.csv")
data


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[4]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[5]:


data1.info()


# In[6]:


data1.isnull().sum()


# In[7]:


data1.describe()


# In[12]:


#boxplot for daily collumn
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"],vert = False)
plt.show()


# In[13]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[17]:


plt.figure(figsize=(6,3))
plt.title("Box plot for sunday Sales")
plt.boxplot(data1["sunday"],vert = False)
plt.show()


# # observation
# -THERE ARE NO MISSING VALUES
# -The daily column values are right skewed 
# -The sunday column values also appear to be right skewed
# -There are two outliers in both daily column and also in sunday column as observed from the boxplots

# # Scatter plot and correlation Strength

# In[18]:


x=data1["daily"]
y=data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.xlim(0, max(y) + 100)
plt.show()


# In[19]:


data1["daily"].corr(data1["sunday"])


# In[20]:


data1[["daily","sunday"]].corr()


# # observations
# -The relationship between x(daily) and y(sunday is seen to be linear as sww from scatter plot
# -The correlation is strong positive with Pearson's correlation od 0.958154

# In[ ]:




