#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


univ = pd.read_csv("Universities.csv")
univ


# In[3]:


univ.describe()
univ


# In[4]:


univ.info()
univ


# In[ ]:




