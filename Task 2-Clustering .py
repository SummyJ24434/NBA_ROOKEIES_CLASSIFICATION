#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing Libaries
import numpy as np #linear algebra
import pandas as pd #data processing, csv file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #MATLAB-Like way of plotting
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


# sklearn package for machine learning in python:
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import preprocessing


# In[3]:


#importing the datasets
address = 'country_data.csv'


# In[4]:


#reading the datasets
df =pd.read_csv(address)
df


# In[5]:


#viewing first ten of the dataset
df.head(10)


# In[6]:


# Lets Visualise the Shape of the Dataset in terms of Rows and Coumns
df.shape


# In[7]:


#adjusting the exports, imports and health datasets because they are in percentage of gdp
df['exports'] = df['exports']*df['gdpp']/100
df['imports'] = df['imports']*df['gdpp']/100
df['health'] = df['health']*df['gdpp']/100


# In[8]:


df


# In[9]:


#checking missing in dataset
df.isnull().sum()


# In[10]:


#description of the dataset
df.describe()


# In[11]:


#datatype
df.info()


# In[12]:


#checking for outliers 
#box plot
# use quantiles to remove the outliers 


# In[13]:


fig = plt.figure(figsize = (12,8))
sns.boxplot(data = df)
plt.show()


# In[14]:


#removing ouliers using quantile
#Q1 = df.quantile(0.25)
#Q3 = df.quantile(0.75)
#IQR = Q3 - Q1

#df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[15]:


#correlation between datasets
df.corr()


# In[16]:


correlation_matrix = df.corr().round(2)
plt.figure(figsize=(9,9))
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True,cmap="RdYlGn")


# From the correlation matrix, the export is highly correlated with GDPP and Import while the GDPP is also highly correlated with Income and health. Child mortality and inflation has high correlation with total fertility

# In[17]:


df.values


# In[18]:


df1 = df.drop("country", axis='columns')
df1


# In[19]:


#Clustering of the Data set


# In[20]:


distortions= []
K = range(1, 10)
#set k number of cluster for range in K
for k in K:
    kmeanModel = KMeans(n_clusters = k)
    kmeanModel.fit(df1)
#adding inertia_ to list
    distortions.append(kmeanModel.inertia_)


# In[21]:


#visualizing distortions
plt.figure(figsize = (10,6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method Showing the optimal K')
plt.show()


# In[22]:


#Tell algorithm the number of clusters it should look for:
kmeans = KMeans(n_clusters = 3)


# In[23]:


X1= Three_features= df[['exports', 'gdpp', 'imports',]]
X1


# Using Three Features

# In[24]:


X1.shape


# In[25]:


scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(X1)
standard_df = pd.DataFrame(standard_df, columns =['exports', 'gdpp', 'imports'])


# In[26]:


#run the Kmeans algorithm for the data X:
kmeans.fit(standard_df)


# In[27]:


#predict which cluster each data point X belongs to:
y_kmeans = kmeans.predict(standard_df)


# In[28]:


y_kmeans.shape


# In[29]:


centers = kmeans.cluster_centers_
#print(centers)
print(centers)


# In[30]:


y_kmeans in kmeans.labels_


# In[31]:


df['cluster_country'] = y_kmeans


# In[112]:


plt.scatter(standard_df['gdpp'], standard_df['exports'], c=df['cluster_country'],cmap= 'rainbow')


# In[33]:


u_labels = np.unique(y_kmeans)


# In[34]:


np.unique(y_kmeans)


# In[35]:


for i in u_labels:
    print(i)


# In[36]:


#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(y_kmeans)
 
#plotting the results:
for i in u_labels:
    plt.scatter(X1.iloc[y_kmeans == i , 0] , X1.iloc[y_kmeans == i , 1], label = i)
plt.xlabel('exports')
plt.ylabel('gdpp')
plt.legend()
plt.show()
fig.savefig('Cluster_plot.png')


# In[37]:


#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(y_kmeans)
 
#plotting the results:
#for i in u_labels:
    #plt.scatter(X2.iloc[y_kmeans == i , 0] , X.iloc[y_kmeans == i , 2],  label = i)
#plt.xlabel('imports')
#plt.ylabel('gdpp')
#plt.legend()
#plt.show()


# In[38]:


np.unique(y_kmeans)


# In[39]:


y_kmeans


# In[40]:


X1['country_clusters']=y_kmeans


# In[41]:


X1.head()


# In[42]:


X1['c'] = df.country


# In[43]:


X1


# In[44]:


cluster1 = X1.loc[X1.country_clusters==0]


# In[45]:


cluster1['c'].unique()


# In[46]:


cluster2 = X1.loc[X1.country_clusters==1]


# In[47]:


cluster2['c'].unique()


# In[48]:


cluster3 = X1.loc[X1.country_clusters==2]


# In[49]:


cluster3['c'].unique()


# In[50]:


X1['country_clusters'].value_counts()


# In[51]:


Country_cluster_exports=pd.DataFrame(X1.groupby(["country_clusters"]).exports.mean())
Country_cluster_gdpp=pd.DataFrame(X1.groupby(["country_clusters"]).gdpp.mean())
Country_cluster_imports=pd.DataFrame(X1.groupby(["country_clusters"]).imports.mean())


# In[52]:


df1 = pd.concat([Country_cluster_exports, Country_cluster_gdpp, Country_cluster_imports], axis=1)


# In[53]:


df1


# In[54]:


X2 = df[['exports', 'imports', 'gdpp', 'health', 'income', 'life_expec']]


# In[55]:


X2


# In[56]:


scaler = preprocessing.StandardScaler()
standard_df2 = scaler.fit_transform(X2)
standard_df2 = pd.DataFrame(standard_df2, columns =['exports', 'gdpp', 'imports', 'health', 'income', 'life_expec'])


# In[57]:


#run the Kmeans algorithm for the data X:
kmeans.fit(standard_df2)


# In[58]:


#predict which cluster each data point X belongs to:
y2_kmeans = kmeans.predict(standard_df2)


# In[59]:


#getting the centroid (centers)
centers2 = kmeans.cluster_centers_
#print(centers)
print(centers2)


# In[60]:


#
y2_kmeans in kmeans.labels_


# In[61]:


u2_labels = np.unique(y2_kmeans)


# In[62]:


#calling the clusters
np.unique(y2_kmeans)


# In[63]:


#confirming the number of clusters used for prediction
for i in u2_labels:
    print(i)


# In[64]:


#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(y_kmeans)
 
#plotting the results:
for i in u2_labels:
    plt.scatter(X2.iloc[y_kmeans == i , 0] , X2.iloc[y_kmeans == i , 1], label = i)


plt.legend()
plt.show()
fig.savefig('Cluster2_plot.png')


# In[65]:


#calling out the clusters generated for the chosen data set
y2_kmeans


# In[66]:


#putting the clusters into the data frame
X2['country_clusters']=y2_kmeans


# In[67]:


#calling the first five rows of the data set
X2.head()


# In[68]:


#adding the country names into the data set with clusters
X2['c'] = df.country


# In[69]:


#checking the countries at clusters=0
group1 = X2.loc[X2.country_clusters==0]


# In[70]:


group1['c'].unique()


# In[71]:


#checking the countries at clusters =1
group2 = X2.loc[X2.country_clusters==1]


# In[72]:


group2['c'].unique()


# In[73]:


#checking the countries at clusters = 2
group3 = X2.loc[X2.country_clusters==2]


# In[74]:


group3['c'].unique()


# In[75]:


X2['country_clusters'].value_counts()


# In[76]:


#evaluating the mean of the clusters
X2.groupby(['country_clusters']).mean()


# In[77]:


#Using all features (adding more features)
X4 = df[['child_mort','exports', 'imports', 'gdpp', 'health', 'income', 'inflation', 'life_expec', 'total_fer']]


# In[78]:


scaler = preprocessing.StandardScaler()
standard_df3 = scaler.fit_transform(X4)
standard_df3 = pd.DataFrame(standard_df3, columns =['child_mort','exports', 'imports', 'gdpp', 'health', 'income', 'inflation', 'life_expec', 'total_fer'])


# In[79]:


#run the Kmeans algorithm for the data X:
kmeans.fit(standard_df3)


# In[80]:


#predict which cluster each data point X belongs to:
y4_kmeans = kmeans.predict(standard_df3)


# In[81]:


#getting the centers of the cluster
centers4 = kmeans.cluster_centers_
#print(centers)
print(centers4)


# In[82]:


u4_labels = np.unique(y4_kmeans)


# In[83]:


for i in u4_labels:
    print(i)


# In[84]:


#putting the clusters into the data set table
X4['country_clusters']=y4_kmeans


# In[85]:


#adding country names to the dataset containing the clusters
X4['c'] = df.country


# In[86]:


#counting the clusters of countries generated
X4['country_clusters'].value_counts()


# In[87]:


#grouping the countries based on the clusters
X4.groupby(['country_clusters']).mean()


# In[88]:


#checking countries in cluster 0
grp1 = X4.loc[X4.country_clusters==0]


# In[89]:


grp1['c'].unique()


# In[90]:


#checking countries in cluster 2
grp2 = X4.loc[X4.country_clusters==1]


# In[91]:


grp2['c'].unique()


# In[92]:


#checking countries in group 3
grp3 = X4.loc[X4.country_clusters==2]


# In[93]:


grp3['c'].unique()


# In[104]:


#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(y4_kmeans)
 
#plotting the results:
for i in u2_labels:
    plt.scatter(X4.iloc[y4_kmeans == i , 0] , X4.iloc[y4_kmeans == i , 1], label = i)
plt.xlabel('c0', fontsize=10)
plt.ylabel('C1', fontsize=10)
plt.title('Country Clusters with Centroid')

plt.legend()
plt.show()
fig.savefig('Cluster3_plot.png')


# In[111]:


#visualizing the clusters
fig = plt.figure(figsize = (12,8))
sns.scatterplot(x='gdpp',y='income',hue='country_clusters',legend='full',data=X4)
plt.xlabel('GDPP', fontsize=10)
plt.ylabel('Income', fontsize=10)
plt.title('GDPP vs Child Mortality')
plt.show()
plt.savefig('clustering3.png')


# In[108]:


#visualizing the clusters
fig = plt.figure(figsize = (12,8))
sns.scatterplot(x='income',y='exports',hue='country_clusters',legend='full',data=X4)
plt.xlabel('Imports', fontsize=10)
plt.ylabel('exports', fontsize=10)
plt.title('GDPP vs  Exports')
plt.show()
plt.savefig('clustering3.png')


# In[113]:


#visualizing the clusters
fig = plt.figure(figsize = (12,8))
sns.scatterplot(x='gdpp',y='child_mort',hue='country_clusters',legend='full',data=X4)
plt.xlabel('GDPP', fontsize=10)
plt.ylabel('Child_mort', fontsize=10)
plt.title('GDPP vs Child Mortality')
plt.show()
plt.savefig('clustering3.png')


# In[ ]:




