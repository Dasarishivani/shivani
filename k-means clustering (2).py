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


# In[5]:


univ=univ.iloc[:,1:]
univ


# In[6]:


cols=univ.columns


# In[7]:


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


scaled_univ_df = pd.DataFrame(scaler.fit_transform(univ), columns=cols)

scaled_univ_df


# In[8]:


from sklearn.cluster import KMeans

# Create a KMeans object with 3 clusters and a random state
clusters_new = KMeans(n_clusters=3, random_state=0)

# Fit the model with the scaled data
clusters_new.fit(scaled_univ_df)


# In[9]:


clusters_new.labels_


# In[10]:


set(clusters_new.labels_)


# In[11]:


univ['clusterid_new'] = clusters_new.labels_
grouped_mean = univ.iloc[:, 1:].groupby("clusterid_new").mean()
grouped_mean


# # observations
# -cluster 2 appers to be the top rated universities cluster as the cutoff score top10 SFRratio parameter values are highest
# cluster 1 apperas to occupy the midde level rated universities
# -cluster 0 comes as the lower level rated universities

# In[12]:


univ[univ['clusterid_new']==0]


# In[13]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


wcss = []


for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, random_state=0)  # Corrected the class reference
    kmeans.fit(scaled_univ_df)
    wcss.append(kmeans.inertia_)  # Inertia represents WCSS (Within-Cluster Sum of Squares)

# Print the WCSS values for each k
print(wcss)

# Plot the WCSS against the number of clusters
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Corrected the typo here
plt.show()


# In[17]:


from sklearn.metrics import silhouette_score
score=silhouette_score(scaled_univ_df,       clusters_new.labels_,metric='euclidean')
score


# In[ ]:




