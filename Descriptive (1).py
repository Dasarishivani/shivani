#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[8]:


df = pd.read_csv("Universities.csv")
df


# In[7]:


np.mean(df["SAT"])


# In[9]:


np.median(df["SAT"])


# In[10]:


df.describe()


# In[11]:


#stantdard deviation of data 
np.std(df["GradRate"])


# In[12]:


#Find the variance
np.var(df["SFRatio"])


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[25]:


sns.histplot(df["Accept"], kde =True)


# In[26]:


plt.figure(figsize=(6,3))
plt.title("GradRate")
plt.hist(df["GradRate"])


# In[30]:


plt.figure(figsize=(6,2))
plt.title("GrandRate")
plt.hist(df["SFRatio"])


# ## observations
#  - In Acceptance ratio the data distribution in non-symmertical and right skewed

# In[ ]:


observation:
    Accept:right skewed
        

