#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
import seaborn as sns


# In[2]:


# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics


# In[3]:


#importing the datasets
address = 'nba_rookie_data.csv'


# In[4]:


#reading the datasets
nba_df =pd.read_csv(address)
nba_df


# In[5]:


# Lets Visualise the Shape of the Dataset in terms of Rows and Coumns
nba_df.shape


# In[7]:


#data type information
nba_df.info()


# In[8]:


#checking missing in dataset
nba_df.isnull().sum()


# In[9]:


#making a copy of the column with empty datasets
nba_new= nba_df[('3 Point Percent')]


# In[10]:


#calling the first five data of the dataset
nba_new.head()


# In[11]:


#filling the data with the mean
nba_new= nba_new.fillna(nba_new.mean())


# In[12]:


#putting back the cleaned column into the main data set
nba_df[('3 Point Percent')]=nba_new


# In[13]:


#checking for missing values again
nba_df.isna().sum()


# In[14]:


#describing the data set
nba_df.describe()


# In[15]:


nba_df.shape


# In[138]:


#correlation between datasets
nba_df.corr()


# In[139]:


#correlation matrix Heatmap
correlation_matrix = nba_df.corr().round(2)
plt.figure(figsize=(11,11))
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True,cmap="RdYlGn")
plt.savefig('nbaplot6.png')


# In[140]:


#grouping and counting the dataset based on the target variable
print(nba_df.groupby('TARGET_5Yrs')['Games Played'].count())


# In[141]:


#defining input and output
x = nba_df.iloc[:, 1].values #input
y = nba_df.iloc[:, -1].values #output


# In[142]:


x = x.reshape(-1, 1)


# In[144]:


# split the data into training and test sets:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3, 
random_state=0)


# In[145]:


#reshaping the dataset
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)


# Logistic Regression

# In[146]:


#Training the model Using logistic regression
logre = LogisticRegression(max_iter=1000)
logre.fit(x_train, y_train)


# In[147]:


#Testing the model
y_pred = logre.predict(x_test)
print('Prediction:', y_pred)


# In[148]:


#checking the accuracy of the model
print('Our Accuracy is %.3f' % logre.score(x_test, y_test))


# In[149]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != logre.predict(x_test)).sum()))


# In[150]:


#visualizing datasets
fig1, ax1= plt.subplots()
ax1.scatter(x_train, y_train, color='blue')
ax1.set_xlabel('Games Played')
ax1.set_ylabel('Target_5Yrs')
ax1.plot(x_test, logre.predict_proba(x_test)[:,1], color='red')
plt.title('Logistic Regression Visualization')
plt.savefig('nbaplot1.png')


# In[151]:


#evaluating the model performance
cm = metrics.confusion_matrix(y_test, y_pred)


# In[152]:


cm #confusion matrix


# In[153]:


#confusion matrix visualization
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(cm, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[154]:


#evaluating model performance using classification report
print(classification_report(y_test,y_pred))


# Gaussian Naive Bayes

# In[155]:


#Training the data with Guassian Naive Bayes Model
gnb = GaussianNB()
gnb.fit(x_train, y_train)


# In[ ]:


#Testing the data set
y_pred = gnb.predict(x_test)
print('Prediction:', y_pred)


# In[156]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != gnb.predict(x_test)).sum()))


# In[157]:


#checking the accuracy of the model
print('Our Accuracy is %.3f' % gnb.score(x_test, y_test))


# In[159]:


#visualizing datasets
fig1, ax1= plt.subplots()
ax1.scatter(x_train, y_train, color='blue')
ax1.set_xlabel('Games Played')
ax1.set_ylabel('Target_5Yrs')
ax1.plot(x_test, gnb.predict_proba(x_test)[:,1], color='red')
plt.title('Gaussian Naive Bayes Visualization')
plt.savefig('nbaplot2.png')


# In[160]:


#evaluating the model performance
cm = metrics.confusion_matrix(y_test, y_pred)


# In[161]:


cm #confusion matrix


# In[162]:


#confusion matrix visualization
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(cm, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[163]:


#evaluating model performance using classification report
print(classification_report(y_test,y_pred))


# Neural Network

# In[164]:


#Neural Network
#Training the data set with MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic", random_state=0, max_iter = 2000)
mlp.fit(x_train, y_train)


# In[165]:


#Testing the model
y_pred = mlp.predict(x_test)
print('Prediction:', y_pred)


# In[166]:


#Accuracy of the model
print('Our Accuracy is %.3f' % mlp.score(x_test, y_test))


# In[167]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != mlp.predict(x_test)).sum()))


# In[168]:


#visualizing datasets
fig1, ax1= plt.subplots()
ax1.scatter(x_train, y_train, color='blue')
ax1.set_xlabel('Games Played')
ax1.set_ylabel('Target_5Yrs')
ax1.plot(x_test, mlp.predict_proba(x_test)[:,1], color='red')
plt.title('Neural Network Visualization')
plt.savefig('nbaplot3.png')


# In[169]:


#evaluating the model performance
cm = metrics.confusion_matrix(y_test, y_pred) #cm= confusion matrix


# In[170]:


#confusion matrix visualization
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(cm, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[171]:


#evaluating model performance using classification report
print(classification_report(y_test,y_pred))


# Using Three Features

# In[172]:


#defining input and output
x = nba_df.iloc[:, [1, 2, 3]].values #input
y = nba_df.iloc[:, -1].values #output


# In[173]:


x


# In[174]:


x.shape


# In[175]:


y


# In[176]:


# split the data into training and test sets:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3, 
random_state=0)


# In[177]:


#Using logistic regression
logre = LogisticRegression(max_iter=1000)
logre.fit(x_train, y_train)


# In[178]:


y_pred = logre.predict(x_test)
print('Prediction:', y_pred)


# In[179]:


#checking the accuracy of the model
print('Our Accuracy is %.2f' % logre.score(x_test, y_test))


# In[180]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != logre.predict(x_test)).sum()))


# In[181]:


#evaluating the model performance
cm = confusion_matrix(y_test, y_pred)


# In[182]:


cm #confusion matrix


# In[183]:


#confusion matrix visualization
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(cm, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[184]:


#evaluating model performance using classification report
print(classification_report(y_test,y_pred))


# In[185]:


#Training the data set with Guassian Naive Bayes Model
gnb = GaussianNB()
gnb.fit(x_train, y_train)


# In[186]:


y_pred = gnb.predict(x_test)
print('Prediction:', y_pred)


# In[187]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != gnb.predict(x_test)).sum()))


# In[188]:


#Accuracy of the model
print('Our Accuracy is %.2f' % gnb.score(x_test, y_test))


# In[189]:


y_test.shape


# In[190]:


#evaluating the model performance using confusion matrix
cm1  = metrics.confusion_matrix(y_test, y_pred)


# In[191]:


cm1 #confusion matrix


# In[192]:


#Visualizing the confusion matrix based on actual and predicted
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(cm1, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[193]:


#evaluating the model performance using the classification report
print(classification_report(y_test,y_pred))


# Neural Network

# In[194]:


#Neural Network
#Training the data set with MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic", random_state=0, max_iter = 2000)
mlp.fit(x_train, y_train)


# In[195]:


#Testing the model
y_pred = mlp.predict(x_test)
print('Prediction:', y_pred)


# In[196]:


#Accuracy of the model
print('Our Accuracy is %.2f' % mlp.score(x_test, y_test))


# In[197]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != mlp.predict(x_test)).sum()))


# In[198]:


#confusion matrix
cm2 = metrics.confusion_matrix(y_test, y_pred)


# In[199]:


cm2 #confusion matrix


# In[200]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(cm2, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[201]:


#classification report
print(classification_report(y_test,y_pred))


# Using Six Features

# In[202]:


#defining input and output
x1 = nba_df.iloc[:, [1, 2, 3, 4, 5, 6]].values #input
y1 = nba_df.iloc[:, -1].values #output


# In[203]:


x1.shape
x1


# In[204]:


# split the data into training and test sets:
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size= 1/3, 
random_state=0)


# In[205]:


#Logistic Regression
logre = LogisticRegression(max_iter=1000)
logre.fit(x1_train, y1_train)


# In[206]:


#predicting the dataset
y1_pred = logre.predict(x1_test)
print('Prediction:', y1_pred)


# In[207]:


#checking the accuracy of the model
print('Our Accuracy is %.3f' % logre.score(x1_test, y1_test))


# In[208]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x1_test.shape[0], (y1_test != logre.predict(x1_test)).sum()))


# In[209]:


#evaluating the model performance
con = metrics.confusion_matrix(y1_test, y1_pred)


# In[210]:


con #confusion matrix


# In[211]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(con, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[212]:


#evaluating model performance using classification report
print(classification_report(y1_test,y1_pred))


# Guassian Naive Bayes

# In[213]:


#Training the data set with Guassian Naive Bayes Model
gnb = GaussianNB()
gnb.fit(x1_train, y1_train)


# In[214]:


y_pred = gnb.predict(x1_test)
print('Prediction:', y1_pred)


# In[215]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y1_test != gnb.predict(x1_test)).sum()))


# In[216]:


#checking the accuracy of the model
print('Our Accuracy is %.3f' % gnb.score(x1_test, y1_test))


# In[217]:


#evaluating the model performance
con_matrix = metrics.confusion_matrix(y1_test, y1_pred)


# In[218]:


con_matrix


# In[219]:


#visualizing the confusion matrix
y1label = ["Actual [>=5yrs]","Actual [<=5yrs]"]
x1label = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(con_matrix, annot=True, xticklabels = x1label, yticklabels = y1label, linecolor='white', linewidths=1)


# In[220]:


#evaluating the model performance with Classification report
print(classification_report(y1_test, y1_pred))


# In[221]:


#Neural Network
#Training the data set with MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic", random_state=0, max_iter = 2000)
mlp.fit(x1_train, y1_train)


# In[222]:


#Testing the model
y1_pred = mlp.predict(x1_test)
print('Prediction:', y1_pred)


# In[223]:


#Accuracy of the model
print('Our Accuracy is %.2f' % mlp.score(x1_test, y1_test))


# In[224]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x1_test.shape[0], (y1_test != mlp.predict(x1_test)).sum()))


# In[225]:


#confusion matrix
cm3 = metrics.confusion_matrix(y1_test, y1_pred)


# In[226]:


cm3 #confusion matrix


# In[227]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(cm3, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[228]:


#classification report
print(classification_report(y_test,y_pred))


# Using All Features

# In[229]:


#defining input and output
x4 = nba_df[ nba_df.columns[1:-1]].values #input
y3 = nba_df.iloc[:, -1].values #output


# In[230]:


x4


# In[231]:


scaler = StandardScaler()
x_new = scaler.fit_transform(x4)
scaled_x4 = pd.DataFrame(x_new,  columns =['Games Played', 'Minutes Played', 'Points Per Game', 'Field Goals Made', 'Field Goals Attempt', 'Field Goal Percent', '3 Point Made', '3 Point Attempt', '3 Point Percent', 'Free Throw Made', 'Free Throw Attempt', 'Free Throw Percent', 'Offensive Rebounds', 'Defensive Rebounds', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers'])


# In[232]:


x3=scaled_x4


# In[233]:


# split the data into training and test sets:
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size= 1/3, 
random_state=0)


# In[234]:


#Using logistic regression
logre = LogisticRegression(max_iter=1000)
logre.fit(x3_train, y3_train)


# In[235]:


#predicting the dataset
y3_pred = logre.predict(x3_test)
print('Prediction:', y3_pred)


# In[236]:


#checking the accuracy of the model
print('Our Accuracy is %.4f' % logre.score(x3_test, y3_test))


# In[237]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x3_test.shape[0], (y3_test != logre.predict(x3_test)).sum()))


# In[238]:


#confusion matrix
cm12 = metrics.confusion_matrix(y3_test, y3_pred)


# In[239]:


cm12 #confusion matrix


# In[240]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(cm12, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[241]:


#classification report
print(classification_report(y3_test,y3_pred))


# In[242]:


#Training the data set with Guassian Naive Bayes Model
gnb = GaussianNB()
gnb.fit(x3_train, y3_train)


# In[243]:


y_pred = logre.predict(x3_test)
print('Prediction:', y3_pred)


# In[244]:


#checking the accuracy of the model
print('Our Accuracy is %.4f' % gnb.score(x3_test, y3_test))


# In[245]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x3_test.shape[0], (y3_test != gnb.predict(x3_test)).sum()))


# In[246]:


#confusion matrix
cm11 = metrics.confusion_matrix(y3_test, y3_pred)


# In[247]:


cm11 #confusion matrix


# In[248]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(cm11, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[249]:


#classification report
print(classification_report(y3_test,y3_pred))


# In[250]:


#Neural Network
#Training the data set with MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic", random_state=0, max_iter = 2000)
mlp.fit(x3_train, y3_train)


# In[251]:


#Testing the model
y3_pred = mlp.predict(x3_test)
print('Prediction:', y3_pred)


# In[252]:


#Accuracy of the model
print('Our Accuracy is %.4f' % mlp.score(x3_test, y3_test))


# In[253]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x3_test.shape[0], (y3_test != mlp.predict(x3_test)).sum()))


# In[254]:


#confusion matrix
cm4 = metrics.confusion_matrix(y3_test, y3_pred)


# In[255]:


cm4 #confusion matrix


# In[256]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(cm4, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[257]:


#classification report
print(classification_report(y3_test,y3_pred))


# In[ ]:





# In[ ]:




