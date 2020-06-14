# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix

dataset = pd.read_csv('iris.csv')

X= dataset.iloc[:,[0,1,2,3]].values
y= dataset.iloc[:,[4]].values



#encoding 
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
y= le.fit_transform(y)

#Spliting dataset into Training set and Test set 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)


from sklearn.svm import SVC
classifier = SVC(kernel= 'rbf', random_state= 0)
classifier.fit(X_train, y_train)

y_pred= classifier.predict(X_test)



from sklearn import metrics
acuracyScoreofSVMmodel = metrics.accuracy_score(y_test, y_pred)




import seaborn as sns
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)