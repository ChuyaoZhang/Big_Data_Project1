#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time # to record time when training
# Load the training sample and the test sample
data1 = pd.read_csv("new_data.csv")
data2 = pd.read_csv("test_set.csv")


# In[2]:


# Preprocess the dataset

# extract the labels from the training dataset and the testing dataset
labels_1 = data1['winner'].values
labels_2 = data2['winner'].values

# remove the attributes that will not effect the result (remove irrelevant features)
# the remain attributes are the features for training
features_1 = data1.drop(['gameId','creationTime','seasonId','winner'], axis=1)
features_2 = data2.drop(['gameId','creationTime','seasonId','winner'], axis=1)
#features_1.head(10)


# In[3]:


from sklearn.tree import DecisionTreeClassifier # import Decision Tree Classifier 
from sklearn.model_selection import train_test_split # import train_test_split function
from sklearn.metrics import accuracy_score #for accuracy calculation 


# In[4]:


# Train Decision Tree Classifer
# create Decision Tree classifer
clf_DT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, min_samples_split=2) 
# let max depth to be 5, to avoid overfitting
 
clf_DT = clf_DT.fit(features_1,labels_1)

# calculate the accuracy of DT model(use the training dataset again)
# predict the label for training dataset
pred_labels_1 = clf_DT.predict(features_1) 
print("Accuracy:",accuracy_score(labels_1, pred_labels_1)) 
#Accuracy: 0.9662179653119337


# In[5]:


#Visualizing Decision Tree
from six import StringIO   
from IPython.display import Image   
from sklearn.tree import export_graphviz 
import pydotplus 
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' 
dot_data = StringIO() 
export_graphviz(clf_DT, out_file=dot_data,   
                filled=True, rounded=True, 
                special_characters=True, 
                feature_names = ['gameDuration','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'],
                class_names=['1','2']) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('DT_figure.png') # save the figure of the tree as .png format
Image(graph.create_png()) # display the figure on the page


# In[6]:


# Find the optimal depth of the DT
# when the depth is increasing, the accuracy for training dataset will increase
# but the accuracy for testing dataset may not always increase
# my solution is to draw the figure of the accuracy with different depth, then find the corresponding depth for max accuracy

# Split training dataset into training set and validation set
# 70% for training and 30% for validation(used to estimate generalization accuracy)
X_train, X_test, Y_train, Y_test = train_test_split(features_1, labels_1, test_size=0.3, random_state=1) 

Accuracy = [] # record the accuracy
for DT_depth in range(1,21): # depth from 1 to 20
    clf_DT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=DT_depth, min_samples_split=2)
    clf_DT = clf_DT.fit(X_train,Y_train)
    pred_X_test = clf_DT.predict(X_test)
    Accuracy.append(accuracy_score(Y_test, pred_X_test))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.plot(range(1,21), Accuracy, label='accuracy')
plt.xlabel('DT_depth') 
plt.ylabel('Accuracy') 
plt.title('The relationship between the depth of DT and the accuracy') 
plt.grid(True) 

# from the figure, we can find that:
# with the increment of the depth, the accuracy will first increase and then decrease
# this result is known as overfitting
# 2 main reasons that cause the overfitting: noise in train set and high complexity of model
# in overfitting, training error is small but test error is large 


# In[7]:


# find the max accuracy and the corresponding depth
max_accuracy = max(Accuracy)
max_index = (Accuracy.index(max_accuracy)+1)
print("The max accuracy of DT is ", max_accuracy, ",and its corresponding depth is ", max_index)
# The max accuracy of DT is  0.9660267471958585 ,and its corresponding depth is 6


# In[8]:


# build the optimal DT(with optimal depth) and evaluate its accuracy
begin_time = time.time() # record the time when training begins
clf_DT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=6, min_samples_split=2) #let depth=6
clf_DT = clf_DT.fit(features_1,labels_1)
end_time = time.time() # record the time when training ends
time_DT = end_time-begin_time #the time for training DT

# predict the label for training dataset
pred_labels_DT_1 = clf_DT.predict(features_1) 
print("Accuracy:",accuracy_score(labels_1, pred_labels_DT_1)) 
print("Training time for DT:", time_DT)
# Accuracy: 0.9685801190784364
# Training time for DT: 0.0625002384185791


# In[9]:


# apart from DT, we can also build other kinds of model

# build MLP model
from sklearn.neural_network import MLPClassifier
begin_time = time.time()
clf_MLP = MLPClassifier(hidden_layer_sizes=(100, ), alpha=0.0001, learning_rate='constant', learning_rate_init=0.01, max_iter=200)
clf_MLP = clf_MLP.fit(features_1,labels_1)
end_time = time.time() 
time_MLP = end_time-begin_time # the time for training MLP

# calculate the accuracy of MLP model
# predict the label for training dataset
pred_labels_MLP_1 = clf_MLP.predict(features_1) 
print("Accuracy:",accuracy_score(labels_1, pred_labels_MLP_1)) 
print("Training time for MLP:", time_MLP)
# Accuracy: 0.9501359047372508
# Training time for MLP: 9.733752012252808


# In[10]:


# build K-NN model
from sklearn.neighbors import KNeighborsClassifier
begin_time = time.time()
clf_KNN = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
clf_KNN = clf_KNN.fit(features_1,labels_1)
end_time = time.time() 
time_KNN = end_time-begin_time # the time for training KNN

# calculate the accuracy of K-NN model
# predict the label for training dataset
pred_labels_KNN_1 = clf_KNN.predict(features_1) 
print("Accuracy:",accuracy_score(labels_1, pred_labels_KNN_1)) 
print("Training time for KNN:", time_KNN)
# Accuracy: 0.9670269220812839
# Training time for KNN: 0.21698927879333496


# In[11]:


# now I have built 3 models(DT,MLP,KNN) for classification, they all have an accuracy higher than 0.9
# each classifier has its own advantages and disadvantages
# the next step is to combine them and attain an ensemble model(minimize the error)

# I choose the voting based ensemble method
# in the ensemble, the output of each classifier will have a weight when voting
# the default weights are set as 1, and the weight will be updated iteratively
# to finish the voting, I build an artificial neural network(ANN), which has 3 layers
# use the labels of 3 models as the input in first layer
# the sum of the product of each input and its weight, is the second layer
# the third layer is the output label, which apply an activation function

#Collect the input

# change label 2 to -1, in order to eliminate the bias in ANN
labels_std = [] # this will be used when finding the optimal weight of ANN
for i in range(len(labels_1)):
    if pred_labels_DT_1[i] == 2:
        pred_labels_DT_1[i] = -1
    if pred_labels_MLP_1[i] == 2:
        pred_labels_MLP_1[i] = -1
    if pred_labels_KNN_1[i] == 2:
        pred_labels_KNN_1[i] = -1
    if labels_1[i] == 1:
        labels_std.append(1)
    if labels_1[i] == 2:
        labels_std.append(-1)


# generate the input labels
input_1 = []
for i in range(len(labels_1)):
    input_1.append([pred_labels_DT_1[i],pred_labels_MLP_1[i],pred_labels_KNN_1[i]])


# In[12]:


W1 = 1
W2 = 1
W3 = 1 #set each weight as 1 (default value)
r = 0.1 #learning rate

for t in range(10000): # iterate for 10000 times to update the weights
    sum = W1*input_1[t][0] + W2*input_1[t][1] + W3*input_1[t][2]
    # activation function: if sum>0, label=1; if sum<0,label=-1
    if sum > 0:
        output = 1
    else:
        output = -1
    W1 = W1 + r*(labels_std[t]-sum)*input_1[t][0]
    W2 = W2 + r*(labels_std[t]-sum)*input_1[t][1]
    W3 = W3 + r*(labels_std[t]-sum)*input_1[t][2]

print("W1 = ", W1, ", W2 = ", W2, ", W3 = ", W3)
#  W1 =  0.3428120352095917 , W2 =  0.189392838308597 , W3 =  0.46785348055601084


# In[13]:


# at last, use the test data to predict the label, and calculate the accuracy of the final prediction

pred_labels_DT_2 = clf_DT.predict(features_2) 
pred_labels_MLP_2 = clf_MLP.predict(features_2) 
pred_labels_KNN_2 = clf_KNN.predict(features_2)
        
for i in range(len(labels_2)):
    if pred_labels_DT_2[i] == 2:
        pred_labels_DT_2[i] = -1
    if pred_labels_MLP_2[i] == 2:
        pred_labels_MLP_2[i] = -1
    if pred_labels_KNN_2[i] == 2:
        pred_labels_KNN_2[i] = -1

input_2 = []# to store the output of three classifiers
for i in range(len(labels_2)):
    input_2.append([pred_labels_DT_2[i],pred_labels_MLP_2[i],pred_labels_KNN_2[i]])
    
# if sum>0, label=1; if sum<0,label=2
pred_labels_ensemble_2 = []
for i in range(len(labels_2)):
    sum = W1*input_2[i][0] + W2*input_2[i][1] + W3*input_2[i][2]
    if sum > 0:
        pred_labels_ensemble_2.append(1)
    else:
        pred_labels_ensemble_2.append(2)

end_time = time.time()
print("Accuracy for test dataset:",accuracy_score(labels_2, pred_labels_ensemble_2)) 
# Accuracy for test data: 0.9645390070921985




