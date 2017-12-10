import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X=np.loadtxt('Word2vecEmbed320.csv',delimiter=",")
XX=np.loadtxt('Textfeatures.csv',delimiter=",")
X1=np.concatenate((X, XX), axis=1)
print(X1.shape)
Y=np.loadtxt("Label.csv",delimiter=",")
Y1=Y[:,0]
Y2=Y[:,1]
Lab=[]
for i in range(0,len(Y)):
    Lab.append(str(int(Y1[i]))+str(int(Y2[i])))
Y=np.asarray(Lab)
le = preprocessing.LabelEncoder()
Y=le.fit_transform(Y)
scal=MinMaxScaler()
X1=scal.fit_transform(X1)
print("Done loading")
'''
clf = LogisticRegression(C=0.1)
print("Logistic Regression")
scores = model_selection.cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
print(np.amax(scores))
scores = model_selection.cross_val_score(clf, X, Y1, cv=10, scoring='accuracy')
print(np.amax(scores))
scores = model_selection.cross_val_score(clf, X, Y2, cv=10, scoring='accuracy')
print(np.amax(scores))

clf = linear_model.SGDClassifier(eta0=0.05,loss='log', learning_rate='optimal',alpha=0.005)
print("SGD")
scores = model_selection.cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
print(np.amax(scores))
scores = model_selection.cross_val_score(clf, X, Y1, cv=10, scoring='accuracy')
print(np.amax(scores))
scores = model_selection.cross_val_score(clf, X, Y2, cv=10, scoring='accuracy')
print(np.amax(scores))


clf = LinearDiscriminantAnalysis()
print("Linear Discriminant")
scores = model_selection.cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
print(np.amax(scores))
scores = model_selection.cross_val_score(clf, X, Y1, cv=10, scoring='accuracy')
print(np.amax(scores))
scores = model_selection.cross_val_score(clf, X, Y2, cv=10, scoring='accuracy')
print(np.amax(scores))

clf = RandomForestClassifier(n_estimators=200, random_state=0)
print("Random Forest")
scores = model_selection.cross_val_score(clf, X1, Y, cv=10, scoring='accuracy')
print(np.amax(scores))
scores = model_selection.cross_val_score(clf, X1, Y1, cv=10, scoring='accuracy')
print(np.amax(scores))
scores = model_selection.cross_val_score(clf, X1, Y2, cv=10, scoring='accuracy')
print(np.amax(scores))

X1=np.concatenate((X, XX[:,0:3]), axis=1)
clf = RandomForestClassifier(n_estimators=200, random_state=0)
print("Random Forest")
scores = model_selection.cross_val_score(clf, X1, Y, cv=10, scoring='accuracy')
print(np.amax(scores))
scores = model_selection.cross_val_score(clf, X1, Y1, cv=10, scoring='accuracy')
print(np.amax(scores))
scores = model_selection.cross_val_score(clf, X1, Y2, cv=10, scoring='accuracy')
print(np.amax(scores))
'''
X1=np.concatenate((X, XX[:,0:2]), axis=1)
clf = RandomForestClassifier(n_estimators=200, random_state=0)
print("Random Forest")
scores = model_selection.cross_val_score(clf, X1, Y, cv=10, scoring='accuracy')
print(np.amax(scores))

X1=np.concatenate((X, XX[:,0:1]), axis=1)
clf = RandomForestClassifier(n_estimators=200, random_state=0)
print("Random Forest")
scores = model_selection.cross_val_score(clf, X1, Y, cv=10, scoring='accuracy')
print(np.amax(scores))

clf = RandomForestClassifier(n_estimators=200, random_state=0)
print("Random Forest")
scores = model_selection.cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
print(np.amax(scores))


