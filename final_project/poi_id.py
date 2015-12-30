# -*- coding: utf-8 -*-
#!/usr/bin/python

import sys
import pickle

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

# The features in the data fall into three major types, namely financial features, email features and POI labels.
# financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
#                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
#                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
# (all units are in US dollars)
# email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
#                  'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']
# (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)
# POI label: [‘poi’] (boolean, represented as integer)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'normalized_total_payments', "fraction_to_poi", "fraction_from_poi"] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_dataset.pop('TOTAL', 0)

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.

    if all_messages == 'NaN' or poi_messages == 'NaN':
        return 0.

    fraction = float(poi_messages)/all_messages

    return fraction

def normalize_data(min_val, max_val, val):
    return (float(val) - min_val)/(float(max_val) - min_val)

def get_min_and_max(data_set, attribute_name):
    all_vals = [data_dict[attribute_name] if data_dict[attribute_name] != 'NaN' else 0 for data_dict in data_set.itervalues()]
    return min(all_vals), max(all_vals)

min_total, max_total = get_min_and_max(my_dataset, "total_payments")

for name, data_point in my_dataset.iteritems():

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi

    total_payments = data_point["total_payments"]
    if total_payments == 'NaN':
        total_payments = 0
    normalized_total_payments = normalize_data(min_total, max_total, total_payments)
    data_point["normalized_total_payments"] = normalized_total_payments

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf = AdaBoostClassifier()
# clf = tree.DecisionTreeClassifier(min_samples_split=40)
# clf = SVC(kernel="rbf", C=10000.)
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

test_classifier(clf, my_dataset, features_list)