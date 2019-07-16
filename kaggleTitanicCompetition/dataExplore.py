import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC  
import numpy as np

df = pd.read_csv('train.csv')
df[['F', 'M']] = pd.get_dummies(df['Sex'])
df = df.drop(['Name', 'Cabin', 'Ticket', 'Sex', 'Embarked'], axis = 1)
meanAge = df['Age'].mean()
df['Age'].fillna(meanAge, inplace = True)

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
testData.to_csv('modifiedTestData.csv')
testingOutput = svclassifier.predict(testData)

outPutData = testData['PassengerId']
survived = pd.Dataframe(testingOutput)

survived.to_csv('survived.csv')
outPutData.to_csv('predictions.csv')
