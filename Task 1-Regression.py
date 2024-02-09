#!/usr/bin/env python
# coding: utf-8

# In[181]:


#importing Library
import numpy as np #linear algebra
import pandas as pd #data processing, csv file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #MATLAB-Like way of plotting
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# In[182]:


# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# In[1]:


#importing the datasets
address = 'houseprice_data.csv'


# In[184]:


#reading the datasets
df =pd.read_csv(address)
df


# In[185]:


df.head(10)


# In[186]:


# Lets Visualise the Shape of the Dataset in terms of Rows and Coumns
df.shape


# In[187]:


#description of the dataset
df.describe()


# In[188]:


#datatype
df.info()


# 
# From the house description, it was observed that 
# -The minimum and maximum price of the house was £7500000 and £7700000, respectively. 
# -The maximum number of bedrooms from the house sold was 33 with 8 maximum number of bathrooms.
# -The minimum sqft_living was 290 and maximum sqft_living was 13540.
# -Majority of the house sold (based on 50% and 75% percentile) had 3 bedrooms and 4 bedrooms with most having sqft_living of 2550 at latituda 47.678 and longitude -122.125

# In[189]:


#correlation between datasets
df.corr()


# In[190]:


#Correlation Matrix HeatMap
correlation_matrix = df.corr().round(2)
plt.figure(figsize=(11,11))
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True,cmap="RdYlGn")


# In[192]:


#checking missing values in dataset
df.isnull().sum()


# In[193]:


#checking column names
df.columns


# In[195]:


#defining target variable (output)
y = df[['price']]


# In[196]:


#Visualizing the independent variables with the target variable
fig1, ax= plt.subplots(2,3,figsize=(16,10))
ax[0,0].scatter(df.iloc[:,2],y,color='blue')
ax[0,0].set_xlabel('bathroom')
ax[0,0].set_ylabel('Price')
ax[0,1].scatter(df.iloc[:,3],y)
ax[0,1].set_xlabel('sqft_living')
ax[0,1].set_ylabel('Price')
ax[0,2].scatter(df.iloc[:,9],y)
ax[0,2].set_xlabel('grade')
ax[0,2].set_ylabel('Price')
ax[1,0].scatter(df.iloc[:,10],y)
ax[1,0].set_xlabel('sqft_above')
ax[1,0].set_ylabel('Price')
ax[1,1].scatter(df.iloc[:,11],y)
ax[1,1].set_xlabel('sqft_living15')
ax[1,1].set_ylabel('Price')


# In[199]:


#defining input and output
x = df.iloc[:, 3].values #input
y = df.iloc[:, 0].values #output


# In[200]:


# split the data into training and test sets:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3, 
random_state=0)


# In[201]:


#reshaping the dataset
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)


# In[202]:


#fit the linear least-square regression line in the dataset
regr= LinearRegression()
regr.fit(x_train, y_train)


# In[203]:


#coefficients
print('coefficients:', regr.coef_)


# In[204]:


#intercept
print ('intercept:', regr.intercept_)


# In[205]:


#mean_squared_error
print ('mean squared error: %.8f' 
       % mean_squared_error(y_test, regr.predict(x_test)))


# In[206]:


#coefficient of determination (model performance evaluatiom)
print ('coefficient of determination: %.2f' 
      
       % r2_score(y_test, regr.predict(x_test)))


# In[207]:


#The model less predicted the house price well with one feature base on the coefficient of determination.


# In[208]:


#visualizing training dataset result
fig1, ax1= plt.subplots()
ax1.scatter(x_train, y_train, color='blue')
ax1.plot(x_train,regr.predict(x_train),color='red')
ax1.set_xlabel('Sqft_living')
ax1.set_ylabel('Price')
plt.title('Visualization of the model')
fig3.tight_layout()
fig3.savefig('houseprice_plot.png')


# In[209]:


#visualizing test dataset result
fig4, ax1= plt.subplots()
ax1.scatter(x_test, y_test, color='blue')
ax1.plot(x_test,regr.predict(x_test),color='red')
ax1.set_xlabel('Sqft_living')
ax1.set_ylabel('Price')
fig4.tight_layout()
fig4.savefig('Houseprice_plot2.png')


# In[210]:


#Including more features


# In[211]:


#defining input and output
x = df[['sqft_living', 'sqft_above', 'grade', 'sqft_living15']].values #input
y = df['price'] #output


# In[212]:


#Scaling the dataset
scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns =['sqft_living', 'sqft_above', 'grade', 'sqft_living15'])


# In[213]:


# split the data into training and test sets:
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size= 1/3, 
random_state=0)


# In[215]:


#fit the linear least-square regression line in the dataset
regr= LinearRegression()
regr.fit(x1_train, y_train)


# In[216]:


#coefficients
print('coefficients:', regr.coef_)


# In[217]:


#intercept
print ('intercept:', regr.intercept_)


# In[218]:


#mean_squared_error
print ('mean squared error: %.8f' 
       % mean_squared_error(y_test, regr.predict(x1_test)))


# In[219]:


#coefficient of determination (model performance evaluatiom)
print ('coefficient of determination: %.2f' 
       % r2_score(y_test, regr.predict(x1_test)))


# In[220]:


#The addition of two more feature variables had slight effect on the model performance. The coefficient of determination increased slightly with decreased mean square error.


# In[221]:


#considering all features


# In[222]:


x = df.iloc[:,1:].values #input
y = df.iloc[:,0].values #output


# In[223]:


# split the data into training and test sets:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3, 
random_state=2)


# In[224]:


#fit the linear least-square regression line in the dataset
regr= LinearRegression()
regr.fit(x_train, y_train)


# In[225]:


#coefficients
print('coefficients:', regr.coef_)


# In[226]:


#intercept
print ('intercept:', regr.intercept_)


# In[227]:


#mean_squared_error
print ('mean squared error: %.8f' 
       % mean_squared_error(y_test, regr.predict(x_test)))


# In[228]:



#coefficient of determination (model performance evaluatiom)
print ('coefficient of determination: %.2f' 
       % r2_score(y_test, regr.predict(x_test)))


# In[229]:


#Considering all features with the house price had greater effect on the model performance. The model had better goodness of fit with more features, that is, better prediction of the house price


# In[ ]:


# Testing the model


# In[231]:


df.sample(3)


# In[273]:


# test_data


# In[274]:


actual_y


# In[275]:


# testing the model prediction
regr.predict(test_data[:, :-1])[0]


# In[ ]:





# In[ ]:




