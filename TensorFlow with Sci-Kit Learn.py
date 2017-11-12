
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd 


# In[50]:


iris = pd.read_csv("Iris.csv")


# In[51]:


iris.head()


# In[52]:


iris["Species"] = iris["Species"].map({"Iris-setosa":0,"Iris-virginica":1,"Iris-versicolor":2})


# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(iris.iloc[:,1:5], iris["Species"], test_size=0.33, random_state=42)


# In[55]:


columns = iris.columns[1:5]


# In[56]:


print(columns)


# In[57]:


import tensorflow as tf


# In[58]:


feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]


# In[59]:


def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape = [labels.size,1])
    return feature_cols,label


# In[60]:


classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,20,10],n_classes = 3)


# In[61]:


classifier.fit(input_fn=lambda: input_fn(X_train,y_train),steps = 1000)


# In[62]:


ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test),steps=1)


# In[63]:


print(ev)


# In[64]:


def input_predict(df):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
    return feature_cols


# In[65]:


pred = classifier.predict_classes(input_fn=lambda: input_predict(X_test))


# In[66]:


pred


# In[67]:


print(list(pred))

