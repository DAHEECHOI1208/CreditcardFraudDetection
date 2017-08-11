
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# #READING THE DATA from kaggle
# 
# https://www.kaggle.com/dalpozz/creditcardfraud
#     

# In[53]:


data=pd.read_csv("creditcard.csv")


# In[54]:


data.head()


# In[39]:


data.shape


# #understanding the target variable

# In[5]:


target_count=pd.value_counts(data['Class'],sort=True).sort_index()


# In[7]:


get_ipython().magic('matplotlib inline')
target_count.plot(kind='bar')


# #analysing the fraud and genuine transaction counts

# In[8]:


count_genuine_transaction=len(data[data["Class"]==0])

count_fraud_transaction=len(data[data["Class"]==1])

normal_transactions_percent=count_genuine_transaction/(count_genuine_transaction+count_fraud_transaction)


# In[12]:


print(normal_transactions_percent)


# In[13]:


fraud_transactions_percent=count_fraud_transaction/(count_genuine_transaction+count_fraud_transaction)


# In[14]:


print(fraud_transactions_percent)


# #visualising the target variable

# In[15]:


fraud_trans=data[data["Class"]==1]


# In[23]:


data.head()


# #creating an id for each row to reference it - just like a customer id

# In[24]:


data.insert(0, 'id', range(0, 0 + len(data)))
data


# In[25]:


data.dtypes


# #dropping the target to extract the features

# In[26]:


data.drop(['Class'],axis = 1, inplace = True)


# In[27]:


X = data.iloc[:].values


# #feature scaling using sklearn

# In[28]:


from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range = (0,1))


# In[29]:


X = sc.fit_transform(X)


# implementing Self organising maps using minisom class

# In[30]:


from minisom import MiniSom


# Setting up the dimensions of the map
# hyper parameters: sigma - kept as default
# learning rate- 0.3 changed from 1
# initializing the weights randomly close to zero
# training the model for 100 iterations     
# 

# In[41]:


som = MiniSom(x = 10, y = 10, input_len = 31, sigma = 1.0, learning_rate = 0.3)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)


# In[42]:


from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)
colorbar()


# mapping the features of the data!

# In[43]:


mappings = som.win_map(X)


# In[44]:


len(mappings)


# mapping the fraud transactions!

# In[45]:


fraud=mappings[(3,1)]
fraud = sc.inverse_transform(fraud)
len(fraud)


# creating the dataframe for the fraud transactions

# In[46]:


data_fraud=pd.DataFrame(fraud)
data_fraud.head()


# referencing the id that is the customer id to identify the fraud customers!

# In[47]:


data_fraud.rename(columns={0: 'id'}, inplace=True)
data_fraud.head()


# In[55]:


data.insert(0, 'id', range(0, 0 + len(data)))
data


# extracting the customers list from data to train the Multi layer perceptron
# 
# now we use supervised deep learning algorithm to predict the probability of fraud 

# In[56]:


customers = data.iloc[:, 1:].values


# creating a new column in the data 'is_fraud' to identify and map the fraud customers 
# in the data and create a dependent variable

# In[57]:


is_fraud = np.zeros(len(data))
for i in range(len(data)):
    if data.iloc[i,0] in fraud:
        is_fraud[i] = 1


# #feature scaling

# In[62]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# #importing the MLP dependencies using keras library and tensorflow as backend

# In[63]:


from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()


# In[64]:




# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', 
                     input_dim = 31))

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[106]:


# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 25, epochs = 20)


# In[107]:


# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)


# The multilayer perceptron using the feature mappings from the Self organising maps 
# 
# is trained and for 20 epochs the accuracy managed by the algorithm is 98.98%.
