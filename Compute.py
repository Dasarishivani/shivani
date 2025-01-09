#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#compute a mean value


# In[2]:


def mean_value(*n):
    sum=0
    counter=0
    for x in n:
        counter = counter +1
        sum += x
    mean = sum/counter
    return mean


# In[7]:


#find the product of given numbers
def product (*n):
    result = 1
    for i in range(len(n)):
        result *= n[i]
    return result 


# In[8]:


product(1,2,3,4)


# In[ ]:




