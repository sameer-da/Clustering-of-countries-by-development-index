#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data= pd.read_excel("World_development_mesurement.xlsx")
data.head(10)


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[7]:


data.isna().sum()


# In[8]:


#Dropping columns with almost 50% null values

data.drop(['Business Tax Rate'],axis=1,inplace=True)
data.drop(['Ease of Business'],axis=1,inplace=True)
data.drop(['Hours to do Tax'],axis=1,inplace=True)
data.columns


# In[9]:


pd.set_option('display.max_rows',None)
data.head(2)


# In[10]:


data.info()


# In[11]:


#Cleaning currency related data

data['Health Exp/Capita']= data['Health Exp/Capita'].astype(str).str.replace('$','')
data['Health Exp/Capita']= data['Health Exp/Capita'].str.replace(',','')
data['Health Exp/Capita']= data['Health Exp/Capita'].astype('float')



data['Tourism Inbound']=data['Tourism Inbound'].astype(str).str.replace('$','')
data['Tourism Inbound']=data['Tourism Inbound'].str.replace(',','')
data['Tourism Inbound']=data['Tourism Inbound'].astype('float')


data['Tourism Outbound']=data['Tourism Outbound'].astype(str).str.replace('$','')
data['Tourism Outbound']=data['Tourism Outbound'].str.replace(',','')
data['Tourism Outbound']=data['Tourism Outbound'].astype('float')



# In[12]:


data.dtypes


# In[13]:


data['GDP']=data['GDP'].astype(str).str.replace('$','')
data['GDP']=data['GDP'].str.replace(',','')
data['GDP']=data['GDP'].astype('float')


# In[14]:


data.GDP.info()


# In[15]:


sns.distplot(data['Life Expectancy Female'],kde=True)


# In[16]:


sns.distplot(data['Life Expectancy Male'],kde=True)


# In[17]:


sns.distplot(data['Mobile Phone Usage'],kde=True)


# In[18]:


sns.distplot(data['Internet Usage'],kde=True)


# In[19]:


sns.distplot(data['Birth Rate'],kde=True)


# In[21]:


#Since the data is skewed for the above columns we fill the null values with median

data['Birth Rate']=data['Birth Rate'].fillna(data['Birth Rate'].median())
data['Internet Usage']=data['Internet Usage'].fillna(data['Internet Usage'].median())
data['Mobile Phone Usage']=data['Mobile Phone Usage'].fillna(data['Mobile Phone Usage'].median())
data['Life Expectancy Male']=data['Life Expectancy Male'].fillna(data['Life Expectancy Male'].median())
data['Life Expectancy Female']=data['Life Expectancy Female'].fillna(data['Life Expectancy Female'].median())


# In[22]:


data.isna().sum()


# In[23]:


sns.distplot(data['Population 0-14'],kde=True)


# In[24]:


sns.distplot(data['Population 15-64'],kde=True)


# In[25]:


sns.distplot(data['Population 65+'],kde=True)


# In[26]:


#filling population 0-15 & 15-64 with mean and 65+ with median

data['Population 0-14']=data['Population 0-14'].fillna(data['Population 0-14'].mean())
data['Population 15-64']=data['Population 15-64'].fillna(data['Population 15-64'].mean())
data['Population 65+']=data['Population 65+'].fillna(data['Population 65+'].median())
data['Population Urban']=data['Population Urban'].fillna(data['Population Urban'].mean)


# In[27]:


data.isna().sum()


# In[28]:


# dropping fields with high number of null values
data.drop(['Lending Interest'],axis=1,inplace=True)
data.drop(['Days to Start Business'],axis=1,inplace=True)
data.drop(['Energy Usage'],axis=1,inplace=True)


# In[29]:


#imputation techinques to fill null values

data['Tourism Outbound'].fillna(method='ffill',inplace=True)
data['Tourism Inbound'].fillna(method='bfill',inplace=True)
data['Health Exp % GDP'].fillna(method='ffill',inplace=True)
data['Health Exp/Capita'].fillna(method='ffill',inplace=True)


# In[30]:


data['Tourism Inbound'].fillna(method='ffill',inplace=True)


# In[31]:


data['Infant Mortality Rate'].fillna(method='ffill',inplace=True)
data['CO2 Emissions'].fillna(method='ffill',inplace=True)
data['GDP'].fillna(data['GDP'].median(),inplace=True)


# In[32]:


data.isna().sum()


# In[33]:


data1=data[['Country','Birth Rate','CO2 Emissions','GDP','Health Exp % GDP','Health Exp/Capita','Infant Mortality Rate',
            'Internet Usage','Life Expectancy Female','Life Expectancy Male','Mobile Phone Usage','Number of Records',
            'Population 0-14','Population 15-64','Population 65+','Population Total','Tourism Inbound','Tourism Outbound']]

data1


# In[34]:


data_iloc = data1.iloc[:,1:]
data_iloc.head(5)


# In[35]:


array = data_iloc.values
array


# In[38]:


from sklearn.preprocessing import scale
norm = scale(array)


# In[39]:


norm


# In[40]:


from sklearn.decomposition import PCA

pca = PCA()
pca_values = pca.fit_transform(norm)
pca_values


# In[41]:


pca.components_


# In[42]:


var = pca.explained_variance_ratio_*100
var


# In[43]:


plt.plot(var)


# In[44]:


#elbow point at 3 so we take first 3 PCA values as fields

final_df=pd.concat([data1['Country'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df


# In[45]:


sns.scatterplot(data=final_df, x='PC1', y='PC2', hue='Country')
plt.show()


# In[46]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[47]:


plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(norm,'complete'))


# In[48]:


# Create Clusters
hclusters=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
hclusters


# In[49]:


y=pd.DataFrame(hclusters.fit_predict(norm),columns=['clustersid'])
y['clustersid'].value_counts()


# In[50]:


final_df['Clusterid']=hclusters.labels_
final_df


# In[51]:


sns.scatterplot(data=final_df,x='PC1',y='PC2',hue='Clusterid')


# In[52]:


sns.scatterplot(data=final_df,x='PC2',y='PC3',hue='Clusterid')


# In[53]:


data1


# In[54]:


import pickle


# In[59]:


pickle_out = open("Project - Clustering (1).pkl","wb")
pickle.dump(hclusters,pickle_out)
pickle_out.close()


# In[60]:


picked_model = pickle.load(open('Project - Clustering (1).pkl','rb'))


# In[61]:


picked_model

