#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets,layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


(X_train,y_train), (X_test,y_test)=datasets.cifar10.load_data()
X_train.shape


# In[3]:


X_test.shape


# In[4]:


X_test


# In[13]:


y_train[:5]


# In[14]:


y_train=y_train.reshape(-1,)
y_train[:5]


# In[11]:


classes=['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# In[15]:


def plot_sample(X,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[16]:


plot_sample(X_train,y_train,0)


# In[18]:


plot_sample(X_train,y_train,4)


# In[21]:


X_train=X_train/255
X_test=X_test/255


# In[25]:


ann=models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(3000,activation='relu'),
    layers.Dense(1000,activation='relu'),
    layers.Dense(10,activation='sigmoid'),
])

ann.compile(optimizer='SGD', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
ann.fit(X_train,y_train,epochs=5)


# In[26]:


ann.evaluate(X_test,y_test)


# In[28]:


from sklearn.metrics import confusion_matrix, classification_report
y_pred=ann.predict(X_test)
y_pred_classes=[np.argmax(element) for element in y_pred]

print('Classification report: \n',classification_report(y_test,y_pred_classes))


# In[29]:


cnn=models.Sequential([
    #cnn
    layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    #dense
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax'),
])


# In[34]:


cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[35]:


cnn.fit(X_train,y_train,epochs=10)


# In[ ]:


# The accuracy after 5 epochs in cnn is ~80% while in ann it was only ~50%
# And after 10 epochs the accuracy is 84%


# In[38]:


y_test=y_test.reshape(-1,)
y_test[:5]


# In[40]:


y_pred=cnn.predict(X_test)
y_pred[:5]


# In[42]:


y_classes=[np.argmax(element) for element in y_pred]
y_classes[:5]


# In[44]:


y_test[:5]


# In[45]:


plot_sample(X_test,y_test,1)


# In[46]:


classes[y_classes[1]]


# In[47]:


plot_sample(X_test,y_test,3)


# In[48]:


classes[y_classes[3]]


# In[49]:


print('Classification report: \n',classification_report(y_test,y_classes))


# In[ ]:




