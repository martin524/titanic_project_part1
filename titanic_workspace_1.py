#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


titanic_data = pd.read_csv('titanic_data.csv', index_col=0)
titanic_data.head()


# In[3]:


def num_missing_values(series):
    bool_series = pd.isnull(series)
    return bool_series.sum()


# In[4]:


print('Column', ' Number of missing values')
print('Survived', num_missing_values(titanic_data['Survived']))
print('Pclass', num_missing_values(titanic_data['Pclass']))
print('Name', num_missing_values(titanic_data['Name']))
print('Sex', num_missing_values(titanic_data['Sex']))
print('Age', num_missing_values(titanic_data['Age']))
print('SibSp', num_missing_values(titanic_data['SibSp']))
print('Parch', num_missing_values(titanic_data['Parch']))
print('Ticket', num_missing_values(titanic_data['Ticket']))
print('Fare', num_missing_values(titanic_data['Fare']))
print('Cabin', num_missing_values(titanic_data['Cabin']))
print('Embarked', num_missing_values(titanic_data['Embarked']))


# In[5]:


#checking whether there is a person recorded more than once, and whether there is one ticket per person
print(titanic_data['Ticket'].is_unique)
print(titanic_data['Name'].is_unique)


# In[6]:


bool_duplicate_tickets = titanic_data.duplicated(subset='Ticket', keep=False) #creating a boolean list to check the number of non-unique tickets
bool_duplicate_tickets.sum() 


# In[7]:


#since the number of non-unique tickets was significant, I created a dataset with just those tickets to invesigate it
duplicate_tickets = titanic_data[bool_duplicate_tickets]
duplicate_tickets.head()


# In[8]:


"""grouping this new datasets with duplicate tickets 
and filling in some of the missing values in cabin according to the ticket they have
i.e. if two people have the same ticket and for one of them Cabin information is missing 
then this will fill it with the Cabin information from the other passenger""" 
pd.options.mode.chained_assignment = None 
duplicate_tickets.Cabin = duplicate_tickets.groupby('Ticket').Cabin.transform(lambda x: x.bfill().ffill())
titanic_data.Cabin = titanic_data.Cabin.fillna(value = duplicate_tickets.Cabin)
num_missing_values(titanic_data['Cabin'])


# In[9]:


#Changing the cabin values to show only whether passengers had a cabin or not
titanic_data['Cabin']=titanic_data['Cabin'].fillna(value = 0)
titanic_data.loc[titanic_data.Cabin !=0, 'Cabin'] = 1


# In[10]:


len(titanic_data['Cabin']) - titanic_data['Cabin'].sum()


# In[11]:


titanic_data.head()


# In[12]:


titanic_data.loc[titanic_data['Fare'].idxmax()]


# In[13]:


titanic_data['Fare'].nlargest(10)


# In[14]:


"""Since the largest Fare value was almost twice the next smaller one, 
and after printing out the the passengers with the same fare values, 
I noticed they had the same ticket, so it is reasonable to assume that 
the Fare value was the price of the Ticket no matter for how many passengers it was meant"""
titanic_data.loc[[28,89,342,439]]


# In[15]:


"""I wanted to know the Ticket price per passenger, so I grouped by 
Ticket and divided the Fare for each passenger 
by the number of passengers in each ticket group"""
titanic_data.Fare = titanic_data.groupby('Ticket').Fare.apply(lambda x: x / len(x))


# In[16]:


#Checking whether the method worked with some of the passengers that had highest Fare previously
print(titanic_data['Fare'].nlargest(10))
print(titanic_data.loc[[28,89,342,439]])


# In[17]:


titanic_data['Fare'].nlargest(10)


# In[18]:


print(titanic_data.loc[[528,378,259,680,738,312,743,119,300]]) # Seemes more reasonable compared the the previous values


# In[19]:


titanic_data['Fare'].nsmallest(20)


# In[20]:


titanic_data.loc[[180,264,272,278,303,414,467,482,598,634,675,733,807,816,823]]


# In[21]:


"""Fare value of 0 means passengers got onto Titanic for free,
and some of them even had 1st class privileges, but given that I
have no information that free tickets were given out I can only
assume that this was mistake in the data, and because they are only 
15 passengers(<0.25%) out of dataset of more than 800 I decided to drop them""" 
titanic_data.drop([180,264,272,278,303,414,467,482,598,634,675,733,807,816,823],inplace=True)
titanic_data['Fare'].nsmallest(20)


# In[22]:


titanic_data['Fare'].describe()


# In[23]:


"""Grouping by port and exploring the differences
in Fare, Age according to the port the price in port
C seemes to be much higher than the other 2, while
there is no significant difference in age mean"""

grouped_by_port = titanic_data.groupby('Embarked')
print(grouped_by_port.Age.apply(num_missing_values))
print(grouped_by_port.Age.mean())
print(grouped_by_port.Fare.mean())
print(grouped_by_port.size())


# In[24]:


"""after grouping by port and class it can be seen 
why the price in port C is way higher(more than 50% of passengers are 1st class), 
and why the price in port Q is so low(72 out of 78 passengers are 3rd class)"""
port_class = titanic_data.groupby(['Embarked', 'Pclass'])
print(port_class.size())


# In[25]:


group_class = titanic_data.groupby('Pclass')
group_class['Fare'].mean() #results are as expected, first class tickets are way more expensive than the other 2


# In[26]:


group_by_sex = titanic_data.groupby('Sex')
print(group_class['Age'].mean())
print(group_by_sex['Age'].mean())
print(group_by_sex.size())
group_by_sex_n_class = titanic_data.groupby(['Sex', 'Pclass'])
print(group_by_sex_n_class['Age'].mean())
print(group_by_sex_n_class.size())


# In[27]:


"""Since there is significant difference between 
the mean age of the groups when grouped by Class and Age 
(ex. 1st class male is way older than 2nd class male or 1st class female etc.), 
I decided to fill the missing values in age by the mean value in each group 
i.e. if a passenger is female and 2nd class than the missing value fro age 
will be filled with the average of this group"""

group_by_sex_n_class.Age = group_by_sex_n_class.Age.transform(lambda x: x.fillna(x.mean()))
titanic_data.Age = titanic_data.Age.fillna(value = group_by_sex_n_class.Age)
num_missing_values(titanic_data['Age'])


# In[28]:


#converting all the age values in integers 
titanic_data['Age'] = titanic_data['Age'].astype(int)
titanic_data.head()


# In[29]:


titanic_data[titanic_data['Embarked'].isnull()] # looking at the last missing values in the dataset


# In[30]:


#Given that 73% of all passenger embarked at S, and 60% of first class passengers embarked at S I will fill these values with S
titanic_data.Embarked = titanic_data.Embarked.fillna(value = 'S')
num_missing_values(titanic_data.Embarked)


# In[37]:


"""8 siblings and a spouse is a bit suspicious, 
but because I don't have any additional information 
I decided to leave it like that"""
titanic_data.describe() #there doesn't seem to be any weird values regarding in any of the columns 


# In[32]:


print('Column', ' Number of missing values')
print('Survived', num_missing_values(titanic_data['Survived']))
print('Pclass', num_missing_values(titanic_data['Pclass']))
print('Name', num_missing_values(titanic_data['Name']))
print('Sex', num_missing_values(titanic_data['Sex']))
print('Age', num_missing_values(titanic_data['Age']))
print('SibSp', num_missing_values(titanic_data['SibSp']))
print('Parch', num_missing_values(titanic_data['Parch']))
print('Ticket', num_missing_values(titanic_data['Ticket']))
print('Fare', num_missing_values(titanic_data['Fare']))
print('Cabin', num_missing_values(titanic_data['Cabin']))
print('Embarked', num_missing_values(titanic_data['Embarked']))


# In[33]:


#dropping columns that I won't be exploring in my analysis 
titanic_data.drop('Name',axis=1, inplace = True) #dropping the name column because each passenger has unique id
titanic_data.drop('Ticket', axis=1, inplace = True) #Used it up until now to clean and fix the data, dropping it becuse I won't use it in my analysis


# In[34]:


#dropping some columns because of high correlation(|corr|>0.75) with ones that I'm more interested in
print(titanic_data['Pclass'].corr(titanic_data['Cabin']))
titanic_data.drop('Cabin', axis=1, inplace = True) # Dropping the cabin column because it is highly correlated to the class of the passengers, so it won't provide any additional info


# In[41]:


"""All of the other correlations bellow are pretty low,
except for the one between Class and Fare, but it is 
still not high enough to be dropped |-0.672|<0.75, 
which I set as a threshold for dropping colums"""

print(titanic_data['Pclass'].corr(titanic_data['SibSp']))
print(titanic_data['SibSp'].corr(titanic_data['Parch']))
print(titanic_data['Age'].corr(titanic_data['SibSp']))
print(titanic_data['Age'].corr(titanic_data['Parch']))
print(titanic_data['Pclass'].corr(titanic_data['Fare']))


# In[36]:


titanic_data.head()


# In[42]:


titanic_data.to_csv('cleaned_titanic_data.csv')

