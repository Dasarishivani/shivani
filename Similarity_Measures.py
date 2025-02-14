#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr


# In[2]:


user1 = np.array([4,5,2,3,4])
user2 = np.array([5,8,7,9,5])


# In[3]:


cosine_similarity = 1 - cosine(user1, user2)
print(f"Cosine Similarity: {cosine_similarity:.4f}")


# In[4]:


#pearson correlation si,ilarity
pearson_corr, _=pearsonr(user1,user2)
print(f"Pearson Correlation Similarity: {pearson_corr:.4f}")


# In[8]:


#euclidean  distance similaruty
euclidean_distance = euclidean(user1,user2)
euclidean_similarity=1/(1+euclidean_distance)
print(f"Euclidean Distance Similarity: {euclidean_similarity:.4f}")


# In[ ]:




