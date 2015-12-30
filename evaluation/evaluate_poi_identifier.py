#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from time import time

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn import tree, cross_validation
from sklearn.metrics import accuracy_score, recall_score, precision_score

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()

t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0,3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1,3), "s"

acc = accuracy_score(pred, labels_test)
print acc

print recall_score(labels_test, pred)
print precision_score(labels_test, pred)
