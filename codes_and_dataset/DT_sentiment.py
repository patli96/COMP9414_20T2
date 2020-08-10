import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree


# read file names from command line
train_file, test_file = sys.argv[1], sys.argv[2]

# preprocess training and test dataset
train_data = pd.read_csv(train_file, delimiter='\t', header=None)
train_data[1].replace(r'https?\:\S+', ' ', regex=True, inplace=True)
train_data[1].replace(r'[^\w #@$%]', '', regex=True, inplace=True)

test_data = pd.read_csv(test_file, delimiter='\t', header=None)
test_data[1].replace(r'https?\:\S+', ' ', regex=True, inplace=True)
test_data[1].replace(r'[^\w #@$%]', '', regex=True, inplace=True)

# create CountVectorizer and fit it with training data
count = CountVectorizer(lowercase=False, max_features=100, token_pattern=r'(?u)\S{2,}')
X_train_bag_of_words = count.fit_transform(train_data.T.values[1])

# get labels for training data
y_train = train_data.T.values[2]

# transform the test data into bag of words created with fit_transform
X_test_bag_of_words = count.transform(test_data.T.values[1])

# get labels for test data
y_test = test_data.T.values[2]
test_index = test_data.T.values[0]

# implement Decision Tree method
clf = tree.DecisionTreeClassifier(min_samples_leaf=0.01, criterion='entropy', random_state=0)
model = clf.fit(X_train_bag_of_words, y_train)
y_pred = clf.predict(X_test_bag_of_words)

# print instance_number and predicted sentiment
for i in range(len(y_pred)):
    print(test_index[i], y_pred[i])

