#!/usr/bin/env python
# coding: utf-8

# ASSUMPTIONS
# 1.Linearity: The relationship between the predictors(X) and the response(Y) is linear.
# 2.Independence: Observations are independent of each other.
# 3.Homoscedasticity: The residuals(Y-Y_hat) exhibit constant variance at all levels of the predictor.
# 4.Normal Distribution Of Errors: The residuals of the model are normally distributed.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars=pd.read_csv("Cars.csv")
cars.head()


# In[3]:


#Rearrange the columns
cars=pd.DataFrame(cars, columns=['HP','VOL','SP','WT','MPG'])
cars.head()

#MPG: Milege of the car(Mile per Gallon)(This is Y-column to be predicted)
#HP: Horse power of the car(X1 column)
#VOL: Volume of the car (size) (X2 column)
#SP: Top speed of the car(Miles per Hour)(X3 column)
#WT: Weight of the car(pounds)(X4 column)


# #EDA

# In[4]:


cars.info()


# Observations:
# - There are no missing values
# - There are 81 observations(81 diffeent car data)
# - The data types of the columns are also releveant and valid

# In[13]:


fig, (ax_box, ax_hist)=plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# obseravtions:
# -There are some extreme values observed in towards the right tail od SP and HP distributions.
# -In VOL and WT columns,a few outliers are observed in both tails of thier distributions.
# -The extreme values of cars data may have come frm the specially designed nature of cars.
# -As this multi-dimensional data,the outliers with respect to spatial dimension may have to be considered while buildin the regression model

# In[11]:


#checking for duplicate rows
cars[cars.duplicated()]


# In[14]:


#pair plots and correlation coefficients
sns.set_style(style="darkgrid")
sns.pairplot(cars)


# In[15]:


cars.corr()


# In[ ]:


#observations:
-Between x and y all the varaiables are showing moderate to high correlation strengths,highest being b/w HP and  MPG
-Therefore the dataset qualifies for building a multiple lijnear regression model to predict MPG
-Ampng x columns (x1,x2,x3anndx4) some very high correlation are observed b/w SP and HP,VOL and WT
_The correrlation among x columns is not  desirable as it ,ight be ;ead to multicollinearity problem


# In[16]:


#build model
#import statmodels.formula.api as smf
model=smf.ols("MPG~WT+VOL+SP+HP",data=cars).fit()


# In[17]:


model.summary()


# In[ ]:




