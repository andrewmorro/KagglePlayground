import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
#from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm

sns.set_style("whitegrid")
#
# class DRModel(Sequential):
#     def __init__(self):
#         super(DRModel, self).__init__()
#         self.conv1 = Conv2D(64,3,activation='relu')

df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
df.info()

X = df.drop("label",axis=1)

X= X/255
y = df['label']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)
test_df = test_df/255

#drmodel = DecisionTreeRegressor(random_state=1)
drmodel = svm.SVC(gamma='auto',cache_size=2000,verbose=True)
drmodel.fit(x_train,y_train)
results=drmodel.score(x_test,y_test)
print(results)

pred = drmodel.predict(test_df)
#seed = 7
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(drmodel, X, y, cv=kfold)
#print(results.mean())

Label = pd.Series(pred,name = 'Label')
ImageId = pd.Series(range(1,28001),name = 'ImageId')
submission = pd.concat([ImageId,Label],axis = 1)
submission.to_csv('submission.csv',index = False)