# 700741122-ASSIGNMENT5-SAI-VARUN-THABETI

Link for the recording: https://drive.google.com/file/d/16DKCEuP889meybNYYHW2f17n_gP00g_z/view?usp=drive_link
Use train_test_split to create training and testing part
Evaluate the model on test part using score and
classification_report(y_true, y_pred)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score


glass_data = pd.read_csv('glass.csv')

x_train = glass_data.drop("Type", axis=1)
y_train = glass_data['Type']


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# Train the model using the training sets
gnb = GaussianNB()
gnb.fit(x_train, y_train)


y_pred = gnb.predict(x_test)
# Classification report 
qual_report = classification_report(y_test, y_pred)
print(qual_report)
print("Naive Bayes accuracy is: ",  (accuracy_score(y_test, y_pred))*100)





2. Implement linear SVM method using scikit library
Use the same dataset above
Use train_test_split to create training and testing part
Evaluate the model on test part using score and
classification_report(y_true, y_pred)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


glass_data = pd.read_csv('glass.csv')

x_train = glass_data.drop("Type", axis=1)
y_train = glass_data['Type']
# splitting train and test data using train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# Train the model using the training sets
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
# Classification report 
qual_report = classification_report(y_test, y_pred, zero_division = 0)
print(qual_report)
print("SVM accuracy is: ", accuracy_score(y_test, y_pred)*100)



Which algorithm you got better accuracy? Can you justify why?
              Gaussian algorithm gives better accuracy. As the accuracy we got upon training on gaussian is
               greater than that of SVM.

