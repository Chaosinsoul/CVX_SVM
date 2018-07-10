
# coding: utf-8

# In[1]:


# Author: Chao Long and Guorui Yang 
import numpy as np
import matplotlib.pyplot as plt
import scipy
from mnist import MNIST
import math
from numpy import linalg as LA
get_ipython().magic(u'matplotlib inline')
mndata = MNIST('./data')
images_training, labels_training = mndata.load_training()
images_testing, labels_testing = mndata.load_testing()  
print(len(images_testing))
print(len(images_training))


# In[2]:


from sklearn.svm import SVC
import sklearn 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from pylab import *
# !pip install statsmodels --user 
# import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt


# In[6]:


training_X = np.matrix(images_training[:10000]).T 
training_Y = np.matrix(labels_training[:10000])
testing_X = np.matrix(images_testing).T 
testing_Y = np.matrix(labels_testing)

# print(training_Y[0,:30])


# fig = plt.figure(figsize=(9,9))
# plt.subplot(2,5,1)
# plt.imshow(training_X[:,1].reshape(28,28))
# plt.title("0")
# plt.subplot(2,5,2)
# plt.imshow(training_X[:,8].reshape(28,28))
# plt.title("1")
# plt.subplot(2,5,3)
# plt.imshow(training_X[:,5].reshape(28,28))
# plt.title("2")
# plt.subplot(2,5,4)
# plt.imshow(training_X[:,7].reshape(28,28))
# plt.title("3")
# plt.subplot(2,5,5)
# plt.imshow(training_X[:,2].reshape(28,28))
# plt.title("4")
# plt.subplot(2,5,6)
# plt.imshow(training_X[:,0].reshape(28,28))
# plt.title("5")
# plt.subplot(2,5,7)
# plt.imshow(training_X[:,13].reshape(28,28))
# plt.title("6")
# plt.subplot(2,5,8)
# plt.imshow(training_X[:,15].reshape(28,28))
# plt.title("7")
# plt.subplot(2,5,9)
# plt.imshow(training_X[:,17].reshape(28,28))
# plt.title("8")
# plt.subplot(2,5,10)
# plt.imshow(training_X[:,4].reshape(28,28))
# plt.title("9")
# plt.show()    
#     
#     
#     
#     

# In[7]:


# prepossing the image data between -1 to 1 
training_X = training_X/127.5-1
testing_X = testing_X/127.5-1


# In[8]:


# data prepossing Assigning the points with label 0 to +1 and the other points within label 1 to 9 to class -1
def Predata(digit,training_Y,testing_Y):
    testing_Y = (testing_Y == digit).astype(np.int)
    testing_Y[testing_Y==0] = -1 
    training_Y = (training_Y == digit).astype(np.int)
    training_Y[training_Y==0] = -1 
    return training_Y,testing_Y


# In[18]:


ticklabel=0
training_Y_,testing_Y_=Predata(ticklabel,training_Y,testing_Y)
# clf = SVC(random_state=0,C=2,kernel="rbf",gamma=0.1)
clf = SVC(C=0.005,kernel="linear")
clf.fit(training_X.T,np.ravel(training_Y_.T))
score = clf.score(testing_X.T,np.ravel(testing_Y_.T))
print("Tetsing accuracy for digit {} is {}".format(i,score))

