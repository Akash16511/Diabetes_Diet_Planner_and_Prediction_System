import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def svm_kernel(input_data):

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


    # Training the Kernel SVM model on the Training set
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    output = classifier.predict(scaler.transform(input_data))
    acc_score = accuracy_score(y_test, y_pred)
    return acc_score , output
