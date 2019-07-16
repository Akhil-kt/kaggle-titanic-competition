import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn import tree

import numpy as np

df = pd.read_csv('train.csv')
df[['F', 'M']] = pd.get_dummies(df['Sex'])
df = df.drop(['Name', 'Cabin', 'Ticket', 'Sex', 'Embarked'], axis = 1)
meanAge = df['Age'].mean()
df['Age'].fillna(meanAge, inplace = True)
df[['1', '2', '3']] = pd.get_dummies(df['Pclass'])
df = df.drop(["Pclass", "PassengerId"], axis=1)


trainingInput = df.drop(['Survived'], axis=1)
trainingOutput = df['Survived']

trainingInput.to_csv('trainingInput.csv')
svclassifier = SVC(kernel='linear')
svclassifier.fit(trainingInput, trainingOutput) 

testData = pd.read_csv('test.csv')
testData[['F', 'M']] = pd.get_dummies(testData['Sex'])
testData = testData.drop(['Name', 'Cabin', 'Ticket', 'Sex', 'Embarked'], axis = 1)
meanAge = testData['Age'].mean()
meanFare = testData['Fare'].mean()
testData['Age'].fillna(round(meanAge), inplace = True)
testData['Fare'].fillna(round(meanFare), inplace = True)
testData[['1', '2', '3']] = pd.get_dummies(testData['Pclass'])
PassengerId = testData["PassengerId"]
testData = testData.drop(["Pclass", "PassengerId"], axis=1)
testData.to_csv('modifiedTestData.csv')
testingOutput = svclassifier.predict(testData)

outPutData = pd.DataFrame(PassengerId)
survived = pd.DataFrame(testingOutput)

outPutData["Survived"] = survived
outPutData.to_csv('predictions.csv')
