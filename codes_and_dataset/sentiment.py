import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from nltk.stem import PorterStemmer


# read file names from command line
train_file, test_file = sys.argv[1], sys.argv[2]

# preprocess training and test dataset
train_data = pd.read_csv(train_file, delimiter='\t', header=None)
train_data[1].replace(r'https?\:\S+', ' ', regex=True, inplace=True)
train_data[1].replace(r'[^a-zA-Z0-9<space>_#@$%]', '', regex=True, inplace=True)

test_data = pd.read_csv(test_file, delimiter='\t', header=None)
test_data[1].replace(r'https?\:\S+', ' ', regex=True, inplace=True)
test_data[1].replace(r'[^a-zA-Z0-9<space>_#@$%]', '', regex=True, inplace=True)

# NLTK Porter stemming
ps = PorterStemmer()
train = []
test = []
for i in range(train_data.shape[0]):
    text = train_data[1][i].split()
    stemmed = [ps.stem(word) for word in text]
    train.append(' '.join(stemmed))
for i in range(test_data.shape[0]):
    text = test_data[1][i].split()
    stemmed = [ps.stem(word) for word in text]
    test.append(' '.join(stemmed))

# create CountVectorizer and fit it with training data
count = CountVectorizer(lowercase=True, max_features=1000, token_pattern=r'(?u)\S{2,}')
X_train_bag_of_words = count.fit_transform(train)

# get labels for training data
y_train = train_data.T.values[2]

# transform the test data into bag of words created with fit_transform
X_test_bag_of_words = count.transform(test)

# get labels for test data
y_test = test_data.T.values[2]
test_index = test_data.T.values[0]

# MLP model
clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(50, ), alpha=0.9655059579094672, learning_rate='invscaling', 
                    learning_rate_init=0.44079441387656165, activation='tanh', max_iter=1000)
model = clf.fit(X_train_bag_of_words, y_train)
y_pred = clf.predict(X_test_bag_of_words)

# print instance_number and predicted sentiment
for i in range(len(y_pred)):
    print(test_index[i], y_pred[i])






