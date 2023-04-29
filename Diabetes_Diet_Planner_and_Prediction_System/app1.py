from decision_tree import decision_tree
from KNN import KNN
from Logisticregression import Logistic_Regression
from naive_bayes import naive_bayes
from randon_forest import random_forest
from svm_kernel import svm_kernel
from SVM import svm



def user(user_choice ,user_input):

    if user_choice == 1:
        acc,output =  Logistic_Regression(user_input)
        return acc, output

    elif user_choice == 2:
        acc,output = svm(user_input)
        return acc, output

    elif user_choice == 3:
        acc,output = svm_kernel(user_input)
        return acc, output

    elif user_choice == 4:
        acc,output = naive_bayes(user_input)
        return acc, output

    elif user_choice == 5:
        acc,output = KNN(user_input)
        return acc, output

    elif user_choice == 6:
        acc,output = decision_tree(user_input)
        return acc, output

    elif user_choice == 7:
        acc,output = random_forest(user_input)
        return acc, output

                                                                                            