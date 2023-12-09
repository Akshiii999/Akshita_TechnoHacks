#!/usr/bin/env python
# coding: utf-8

# # Task 1: Data Cleaning Titanic Dataset

# In[47]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
df=pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Akshita Arora\Practice Data\Project 1 - Sales Data Analysis\Technohack\train.csv")


# In[48]:


df.head()


# In[49]:


df.columns


# In[50]:


df.dtypes #identifying categorical and  continuous variables


# In[51]:


df.describe()     #summary statistics for continuous variables


# In[52]:


df.isnull()         #checking for missing values


# In[53]:


df.isnull().sum()              


# In[54]:


df.dropna()          #dropping rows where there are missing values


# In[55]:


df.dropna().isnull().sum()


# In[56]:


df.dropna(how='all')                #remove MV from rows


# In[57]:


df.dropna(how='all').shape


# In[58]:


df.dropna(axis=1)             #removing MV from columns


# In[59]:


df


# In[60]:


df.drop(columns=['Age','Cabin'])             # dropping columns which are of no use


# In[61]:


df.dropna(axis=1,how='all')          # removing columns where there are MV


# In[62]:


df.fillna(0)                 #filling all MV with 0


# In[63]:


df['Age'].fillna(0)


# In[64]:


df['Age'].fillna(df['Age'].mean())


# # Univariate outlier Detection

# In[65]:


df['Age'].plot.box() 


# In[66]:


df.loc[df['Age']>65,'Age']=np.mean(df['Age'])


# In[67]:


plt.subplot(2,2,1)                        #Outlier detection
plt.boxplot(df.Pclass)
plt.title('Pclass')
plt.show()

plt.subplot(2,2,2)
plt.boxplot(df.PassengerId)
plt.title('PassengerId')
plt.show()


# In[68]:


plt.subplot(3,2,1)
plt.boxplot(df.SibSp)
plt.title('SibpSp')
plt.show()

plt.subplot(3,2,3)
plt.boxplot(df.Parch)
plt.title('Parch')
plt.show()


# In[69]:


plt.boxplot(df.Fare)
plt.title('Fare')
plt.show()


# In[70]:


plt.boxplot(df.Survived)
plt.title('Survived')
plt.show()


# In[71]:


q1=df['Fare'].quantile(0.25)
q3=df['Fare'].quantile(0.75)
IQR=q3-q1

lower_bound=q1-1.5*IQR
upper_bound=q3+1.5*IQR

df=df[(df['Fare']>=lower_bound)&(df['Fare']<=upper_bound)]


# In[72]:


print(q1)
print(q3)
print(IQR)
df


# In[73]:


plt.boxplot(df.Fare)
plt.title('Fare')
plt.show()


# In[74]:


q1=df['Age'].quantile(0.25)
q3=df['Age'].quantile(0.75)
IQR=q3-q1

lower_bound=q1-1.5*IQR
upper_bound=q3+1.5*IQR

df=df[(df['Age']>=lower_bound)&(df['Age']<=upper_bound)]


# In[75]:


print(q1)
print(q3)
print(IQR)
df


# In[76]:


plt.boxplot(df.Age)
plt.title('Age')
plt.show()


# In[90]:


q1=df['Parch'].quantile(0.25)
q3=df['Parch'].quantile(0.75)
IQR=q3-q1

lower_bound=q1-1.5*IQR
upper_bound=q3+1.5*IQR

df=df[(df['Parch']>=lower_bound)&(df['Parch']<=upper_bound)]


# In[91]:


print(q1)
print(q3)
print(IQR)
df


# In[92]:


plt.boxplot(df.Parch)
plt.title('Parch')
plt.show()


# In[80]:


q1=df['SibSp'].quantile(0.25)                 #removing outlier
q3=df['SibSp'].quantile(0.75) 
IQR=q3-q1

lower_bound=q1-1.5*IQR
upper_bound=q3+1.5*IQR

df=df[(df['SibSp']>=lower_bound)&(df['SibSp']<=upper_bound)]


# In[81]:


print(q1)
print(q3)
print(IQR)
df


# In[82]:


plt.boxplot(df.SibSp)
plt.title('SibSp')
plt.show()


# In[ ]:





# In[ ]:




