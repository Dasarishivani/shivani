#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("Universities.csv")
df


# In[4]:


np.mean(df["SAT"])


# In[5]:


np.median(df["SAT"])


# In[6]:


np.std(df["GradRate"])


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[9]:


sns.histplot(df["Accept"], kde =True)


# In[10]:


plt.figure(figsize=(6,3))
plt.title("GradRate")
plt.hist(df["GradRate"])


# In[13]:


#create a pandas series of batmans1 scores
s1 = [20,15,10,25,35,28,40,45]
scores1 = pd.Series(s1)
scores1


# In[14]:


plt.boxplot(scores1, vert=False)


# In[15]:


s1 = [20,15,10,25,35,28,40,45,120,130]
scores1 = pd.Series(s1)
scores1


# In[16]:


plt.boxplot(scores1, vert=False)


# In[17]:


import pandas as pd
import numpy as np


# In[18]:


df = pd.read_csv("Universities.csv")
df


# In[19]:


plt.boxplot(scores1, vert=False)


# In[ ]:


observation:
Accept and SFratio and SAT has highratio value    

