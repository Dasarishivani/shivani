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


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[7]:


#drop unecessary coloumns.


# In[8]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[9]:


#convert the month datatype to integer data type
data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[10]:


#checking duplicate rows
data1[data1.duplicated()]


# In[11]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[12]:


#change the name
data1.rename({'Solar.R':'Solar','Temp':'Temperature'}, axis=1, inplace = True)
data1


# In[13]:


#display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[14]:


#visualize data1 missing values using graph
cols = data1.columns
colours = ['black','yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[15]:


#find the mean and median values
#impitation of missing values with median
median_ozone=data1['Ozone'].median()
mean_ozone=data1['Ozone'].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[16]:


#Replace the ozone missing values with median
data1['Ozone'] = data1["Ozone"].fillna(median_ozone)
data1.isnull().sum()


# In[17]:


#Replace the ozone missing values with mean
data1['Ozone'] = data1["Ozone"].fillna(mean_ozone)
data1.isnull().sum()


# In[19]:


#Find the mode values of categorical column(weather)
print(data1["Weather"].value_counts())
mode_weather=data1["Weather"].mode()[0]
print(mode_weather)


# In[20]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[21]:


data1.tail()


# In[22]:


data1.reset_index(drop=True)


# In[23]:


#create a figure with two subplots,stacjed certically

fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

#plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

#plot the hostogram with kde curve in the second subplot
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color="purple", bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("ozone levels")
axes[1].set_ylabel("Frequency")

#adjusr layout for better spacing
plt.tight_layout
plt.show()


# In[24]:


#OBSERVATIONS

-The ozone column has extreme values 81 as seen from boxplot
-The same is confirmed from the below right-Skewed histogram


# In[25]:


#create a figure for violin plot
sns.violinplot(data=data1["Ozone"], color='red')
plt.title('Violin Plot')


# In[26]:


#create a figure with two subplots,stacjed certically

fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

#plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

#plot the hostogram with kde curve in the second subplot
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color="purple", bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("solarlevels")
axes[1].set_ylabel("Frequency")

#adjusr layout for better spacing
plt.tight_layout
plt.show()


# In[27]:


#Extract outliersthe boxplot for ozone
plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[28]:


data1["Ozone"].describe()


# In[29]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if ((x<(mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# # Qunatitle-Quantile plot for detection

# In[30]:


import scipy.stats as stats
#create Q_Q plot
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for outlier DEtection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# # Observation from Q-Q plot
# -The data does not follow normal distribution data points are deviating significantly away from the red line
# -The data shows a right skewed distribution and possible otliers

# In[31]:


#create a figure for viloin plot
sns.violinplot(data=data1["Ozone"], color='lightpink')
plt.title("Violin Plot")
#show the plot
plt.show()


# In[32]:


sns.swarmplot(data=data1, x = "Weather", y ="Ozone",color="orange",size=6)


# In[33]:


sns.stripplot(data=data1, x = "Weather", y="Ozone",color="pink",size=6,jitter=True)


# In[34]:


sns.kdeplot(data=data1["Ozone"],fill=True, color="yellow")
sns.rugplot(data=data1["Ozone"], color="black")


# In[35]:


sns.boxplot(data=data1, x ="Weather", y="Ozone")


# In[36]:


plt.scatter(data1["Wind"], data1["Temperature"])


# In[37]:


data1["Wind"].corr(data1["Temperature"])


# # observation
# -it is mild negative

# # Observation
# -the correlation between wide and temp is observed to be negatively correlated

# In[40]:


data1.info()


# In[41]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[ ]:


data1_numeric.corr()


# # Observations
# -the highest correlation strength is observed Ozone and Temperature(0.597087)
# -the next highest correlation strength is observed Ozone and wind(-0.523738)
# -the next highest correlation strength is observed between wind and temp(-0.441228)
# -the least correlation strength is observed between solar and wind(0.055874)
# 

# In[42]:


#plot the pair plot btw all numeric columns
sns.pairplot(data1_numeric)


# In[43]:


#creating dummy variable for weather column
data2=pd.get_dummies(data1,columns=['Month','Weather'])
data2


# In[44]:


data1_numeric


# In[45]:


data1_numeric.values


# In[46]:


from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
array=data1_numeric.values
scaler=MinMaxScaler(feature_range=(0,1))
rescaledx=scaler.fit_transform(array)
set_printoptions(precision=2)
print(rescaledx[0:10,:])


# In[49]:


from sklearn.preprocessing import StandardScaler
array=data1_numeric.values
scaler=StandardScaler()
rescaled=scaler.fit_transform(array)
set_printoptions(precision=2)
print(rescaled[0:10,:])


# In[ ]:





# In[ ]:




