import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('C:/Users/dev/OneDrive/Documents/Python Project/DeployingMLModel/Heart_Disease_Prediction.csv')

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
print(heart_data.describe())

# checking the distribution of Target Variable
heart_data['Heart Disease'].value_counts()

X = heart_data.drop(columns='Heart Disease', axis=1)
Y = heart_data['Heart Disease']
'''print(X)
print(Y)'''

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

X.shape, X_train.shape, X_test.shape

#Model Training
model = LogisticRegression()
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

#Model Evaluation
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)
print()

input_data = (65,1,4,120,177,0,0,140,0,0.4,1,0,7)
input_data1 = (56,1,3,130,256,1,2,142,1,0.6,2,1,6)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0] == 'Absence'):
    print('The person dose not have heart disease')
else:
    print('The person have a heart disease')