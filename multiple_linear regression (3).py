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

# In[5]:


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

# In[6]:


#checking for duplicate rows
cars[cars.duplicated()]


# In[7]:


#pair plots and correlation coefficients
sns.set_style(style="darkgrid")
sns.pairplot(cars)


# In[8]:


cars.corr()


# In[9]:


#observations:
-Between x and y all the varaiables are showing moderate to high correlation strengths,highest being b/w HP and  MPG
-Therefore the dataset qualifies for building a multiple lijnear regression model to predict MPG
-Ampng x columns (x1,x2,x3anndx4) some very high correlation are observed b/w SP and HP,VOL and WT
_The correrlation among x columns is not  desirable as it ,ight be ;ead to multicollinearity problem


# In[ ]:


#build model
#import statmodels.formula.api as smf
model=smf.ols("MPG~WT+VOL+SP+HP",data=cars).fit()


# In[ ]:


model.summary()


# # preparing a preliminary model considering all x columns

# In[ ]:


#build model
import statsmodels.formula.api as smf
model1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[ ]:


model1.summary()


# In[ ]:


#observations
-The R-squared and adjusted squared values are good and about 75% of variability y is explained by x columns
-The probability value with respect to f statstic is close to zero,indicating that all are some of x columns are sugnificant
-The p-values for VOL and WT are higher than 5% some interactions issue all among themselves among themselves which need to further explored


# In[ ]:


#performance metrics for model
#find the performance metrics
#create a dataframe with actual y and predicted y columns
df1=pd.DataFrame()
df1["actual_y1"]=cars["MPG"]
df1.head()


# In[ ]:


#predict for the given x data columns
pred_y1=model.predict(cars.iloc[:,0:4])
df1["pred_y1"]=pred_y1
df1.head()


# In[ ]:


cars=pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE :", mean_squared_error(df1["actual_y1"], df1["pred_y1"]))


# In[ ]:


#Compute the MSE(Mean Squared Error) for model1
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse=mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse)
print("RMSE :", np.sqrt(mse))


# # Checking for mulitilinearity among x-columns using VIF method

# In[ ]:


#cars.head()


# In[ ]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[ ]:


#observations
-The ideal range of VIF shall be between 0 to 10.However sightly higher values ca be tolerated
-As seen from the evry high VIF values for VOL and WT,it is clear that they are prone to multicollinearity 
-Hence it is decided to drop one of the columns(either VOL or WT )to overcome the multicollinearity
-it is decided to drop WT and retain VOL column in further models


# In[10]:


cars1 = cars.drop("WT",axis=1)
cars1.head()


# In[11]:


#BUILD MODEL
#import statsmodels.formula.api as smf
model2=smf.ols("MPG~VOL+SP+HP",data=cars1).fit()


# In[12]:


model2.summary()


# In[13]:


df2=pd.DataFrame()
df2["actual_y2"]=cars["MPG"]
df2.head()


# In[14]:


pred_y2=model2.predict(cars.iloc[:,0:4])
df2["pred_y2"]=pred_y2
df2.head


# In[15]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :", np.sqrt(mse))


# # observations
# -The adjusted R-suared value improved slightly to 0.76
# -All the p-values for models parameters are less than 5% henace they are significant
# -Therefore the hP,VOL,SP columns are finalized as the significant predictor for the MPG response variable
# -There is no improvement in MSE value

# # Identification of High Influence points(spatial outliers)

# In[16]:


cars1.shape


# In[ ]:


#### Leverage (Hat Values):
Leverage values diagnose if a data point has an extreme value in terms of the independent variables. A point with high leverage has a great ability to influence the regression line. The threshold for considering a point as having high leverage is typically set at 3(k+1)/n, where k is the number of predictors and n is the sample size.


# In[17]:


#define variables and assign values
k=3
n=81
leverage_cutoff=3*((k+1)/n)
leverage_cutoff


# In[23]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model2,alpha=.05)
y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# In[ ]:


#observations
-From the above plot, it is evident that data data points 65,70,76,78,79,80 are the influencers. as their H Leverage values are higher and size is higher


# In[31]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[38]:


cars2=cars.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)
cars2


# In[42]:


#Rebuild the model model
model3=smf.ols('MPG~VOL+SP+HP',data=cars2).fit()
model3.summary()


# In[43]:


#Performance Metrics for models
df3=pd.DataFrame()
df3["actual_y3"]=cars2["MPG"]
df3.head()


# In[ ]:


#### Comparison of models
                     

| Metric         | Model 1 | Model 2 | Model 3 |
|----------------|---------|---------|---------|
| R-squared      | 0.771   | 0.770   | 0.885   |
| Adj. R-squared | 0.758   | 0.761   | 0.880   |
| MSE            | 18.89   | 18.91   | 8.68    |
| RMSE           | 4.34    | 4.34    | 2.94    |


- **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

