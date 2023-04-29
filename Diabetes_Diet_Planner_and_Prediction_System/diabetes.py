import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

diabetes_data = pd.read_csv("diabetes.csv")

## Independent and Dependent features
x = diabetes_data.drop(columns = ['Outcome'], axis =1)
y = diabetes_data['Outcome']


##Train Test Split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=.2, random_state=2)

## Standardize the dataset
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

pickle.dump(scaler,open('scaling.pkl','wb'))


model = svm.SVC(kernel='linear')
model.fit(x_train,y_train)

### Prediction With Test Data
x_train_prediction = model.predict(x_train)

## Standardize the dataset
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

pickle.dump(scaler,open('scaling.pkl','wb'))


model = svm.SVC(kernel='linear')
model.fit(x_train,y_train)

### Prediction With Test Data
x_train_prediction = model.predict(x_train)


##transformation of new data
input_data = (2,22,22,22,1,1,1,5)
input_data_array = np.asarray(input_data)
input_data_reshape = input_data_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshape)



pickle.dump(model,open('svmodel.pkl' , 'wb'))
pickled_model=pickle.load(open('svmodel.pkl','rb'))