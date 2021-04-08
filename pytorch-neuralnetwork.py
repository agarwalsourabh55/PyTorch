#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[4]:


df =pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df


# In[5]:


import torch 
import seaborn as sns 


# In[6]:


X= df.drop('Outcome',axis=1).values ##independent features
y=df['Outcome'].values


# In[7]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=0)


# In[17]:


### Libraries from Pytorch
import torch 
import torch.nn as nn
import torch.nn.functional as F


# In[18]:


#########CREATING TENSORS

X_train =torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)

y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)



# In[19]:


#Creating model with Pytorch

class ANN_Model(nn.Module):
    def __init__(self,input_features=8,hidden1=20,hidden2=20,
                 out_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
        
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x
    
    


# In[20]:


### INSTANTIATE MY ANN MODEL
torch.manual_seed(20)
model=ANN_Model()


# In[21]:


model.parameters


# In[22]:


##BACKWARD PROPAGATION

#Define loss func and define optimizer 

loss_function = nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.01)



# In[23]:


epochs=500
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model.forward(X_train)
    loss=loss_function(y_pred,y_train)
    final_losses.append(loss)
    
    if i%10==1:
        print("epoch {} and loss{}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[24]:


##plotting the loss func

import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')


# In[25]:


plt.plot(range(epochs),final_losses)
plt.ylabel('Loss')
plt.xlabel('epoc')


# In[26]:


#### prediciton in X_test data
prediction =[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)
        prediction.append(y_pred.argmax().item())


# In[27]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,prediction)
cm


# In[28]:


plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')



# In[29]:


from sklearn.metrics import accuracy_score
score =accuracy_score(y_test,prediction)
score


# In[30]:


torch.save(model,'diabetes.pt')
model= torch.load('diabetes.pt')


# In[31]:


model.eval()


# In[32]:


list(df.iloc[0,:-1])


# In[33]:


lst1=[6.0, 138.0, 722.0, 33.0, 40.0, 33.6, 0.627, 50.0]


# In[34]:


new_data=torch.tensor(lst1)


# In[35]:


with torch.no_grad():
    print(model(new_data))
    print(model(new_data).argmax().item())
    


# In[ ]:




