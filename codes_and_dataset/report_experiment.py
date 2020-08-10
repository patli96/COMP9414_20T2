import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from yellowbrick.text import FreqDistVisualizer


def pre_process(file_name):
    '''
        Pre-processing the given tweets and visualize the top 50 frequent tokens
    '''
    stop_words = set(stopwords.words('english'))
    data = pd.read_csv(file_name, delimiter='\t', header=None).to_numpy()
    ps = PorterStemmer()
    for i in range(data.shape[0]):
        text = data[i][1].split()
        for j in range(len(text)):
            if re.match(r'(http(s*)://.+)', text[j]) is not None:
                text[j] = ' '
            else:
                text[j] = re.sub(r'[^a-zA-Z0-9#@_$%\s]', '', text[j])
        for word in text:
            if word == ' ' or len(word) < 2:
                text.remove(word)
        # stop words removal
        # filtered_data = [word.lower() for word in text if word not in stop_words]
        filtered_data = text
        # NLTK Porter stemming
        for j in range(len(filtered_data)):
            filtered_data[j] = ps.stem(filtered_data[j])
        data[i][1] = ' '.join(filtered_data)

    # create CountVectorizer and fit it with training data
    count = CountVectorizer(lowercase=True, token_pattern=r'\S{2,}')
    x_train_bag_of_words = count.fit_transform(data[:4000, 1])
    y_train = data[:4000, 2]
    # transform the test data into bag of words created with fit_transform
    x_test_bag_of_words = count.transform(data[4000:, 1])
    y_test = data[4000:, 2]
    # visualize the frequency distribution
    features = count.get_feature_names()
    visualizer = FreqDistVisualizer(features=features, orient='h', n=50)
    visualizer.fit(x_train_bag_of_words)
    visualizer.show()
    return x_train_bag_of_words, y_train, x_test_bag_of_words, y_test, data



def vader_baseline(data, y_test):
    '''
        A VADER model is implemented as a baseline.
        This function includes predicting with VADER and evaluating the result.
    '''
    vader_pred = []
    analyser = SentimentIntensityAnalyzer()
    for text in data[4000:, 1]:
        score = analyser.polarity_scores(text)
        if score['compound'] >= 0.05:
            # print(text+": "+"VADER positive")
            vader_pred.append('positive')
        elif score['compound'] <= -0.05:
            # print(text+": "+"VADER negative")
            vader_pred.append('negative')
        else:
            # print(text+": "+"VADER neutral")
            vader_pred.append('neutral')
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_lb = lb.transform(y_test)
    y_pre_lb = lb.transform(vader_pred)
    print('------- VADER')
    print(f'accuracy:   {accuracy_score(y_test, vader_pred)}')
    print(f"auc score:  {roc_auc_score(y_test_lb, y_pre_lb, average='macro')}")
    print(f"classification report:\n{classification_report(y_test,vader_pred, digits=3)}\n")


def random_forest(x_train, y_train, x_test, y_test):
    '''
        Random Forest model and its hyperparameter tuning
    '''
    clf = RandomForestClassifier(bootstrap=False, criterion='entropy', max_features='auto', n_estimators=142,
                                 warm_start=True)
    param_dict = {
        'n_estimators': randint(1, 200),
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'warm_start': [True, False]
    }
    print('------- Random Forest')
    rf_cv = RandomizedSearchCV(clf, param_dict, cv=5, scoring='accuracy')
    rf_cv.fit(x_train, y_train)
    print(f'Tuned MLP Parameters: {rf_cv.best_params_}')
    print(f'Best score is {rf_cv.best_score_}')
    model = clf.fit(x_train, y_train)
    predict_and_evaluate(model, x_test, y_test)


def mlp(x_train, y_train, x_test, y_test):
    '''
        MLP and its hyperparameters tuning
    '''
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(50,), alpha=0.9655059579094672, learning_rate='invscaling',
                        learning_rate_init=0.44079441387656165, activation='tanh', max_iter=1000)
    param_dist_1 = {'solver': ['lbfgs', 'sgd', 'adam'],
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'hidden_layer_sizes': [(100,), (100, 100), (50,), (50, 50, 50), (20,), (20, 20), (20, 20, 20)],
                    'alpha': uniform(loc=1e-7, scale=1)
                    }

    param_dist_2 = {'solver': ['sgd', 'adam'],
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'hidden_layer_sizes': [(100,), (100, 100), (50,), (50, 50, 50), (20,), (20, 20), (20, 20, 20)],
                    'alpha': uniform(loc=1e-7, scale=1),
                    'learning_rate_init': uniform(loc=1e-7, scale=1)
                    }

    param_dist_3 = {'solver': ['sgd'],
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'hidden_layer_sizes': [(100,), (100, 100), (50,), (50, 50, 50), (20,), (20, 20), (20, 20, 20)],
                    'alpha': uniform(loc=1e-7, scale=1),
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'learning_rate_init': uniform(loc=1e-7, scale=1)
                    }
    print('------- MLP')
    print('param_dist_1:')
    mlp_cv = RandomizedSearchCV(clf, param_dist_1, cv=5, scoring='accuracy')
    mlp_cv.fit(x_train, y_train)
    print(f'Tuned MLP Parameters: {mlp_cv.best_params_}')
    print(f'Best score is {mlp_cv.best_score_}')
    print('param_dist_2:')
    mlp_cv = RandomizedSearchCV(clf, param_dist_2, cv=5, scoring='accuracy')
    mlp_cv.fit(x_train, y_train)
    print(f'Tuned MLP Parameters: {mlp_cv.best_params_}')
    print(f'Best score is {mlp_cv.best_score_}')
    print('param_dist_3:')
    mlp_cv = RandomizedSearchCV(clf, param_dist_3, cv=5, scoring='accuracy')
    mlp_cv.fit(x_train, y_train)
    print(f'Tuned MLP Parameters: {mlp_cv.best_params_}')
    print(f'Best score is {mlp_cv.best_score_}')
    model = clf.fit(x_train, y_train)
    predict_and_evaluate(model, x_test, y_test)


def sgd(x_train, y_train, x_test, y_test):
    '''
        SGD and its hyperparameters tuning
    '''
    clf = SGDClassifier(eta0=0.007870258748593513, learning_rate='invscaling', loss='squared_hinge', penalty='l2')
    param_dict = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber',
                           'epsilon_insensitive', 'squared_epsilon_insensitive'],
                  'penalty': ['l2', 'l1', 'elasticnet'],
                  'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                  'warm_start': [True, False],
                  'average': [True, False],
                  'shuffle': [True, False],
                  'eta0': uniform(loc=1e-7, scale=1e-2)}
    sgd_cv = RandomizedSearchCV(clf, param_dict, cv=5, scoring='accuracy')
    sgd_cv.fit(x_train, y_train)
    print('------- SGD')
    print(f'Tuned SGD Parameters: {sgd_cv.best_params_}')
    print(f'Best score is {sgd_cv.best_score_}')
    model = clf.fit(x_train, y_train)
    predict_and_evaluate(model, x_test, y_test)


def predict_and_evaluate(model, x_test, y_test):
    '''
        This function is for predicting and evaluating the result.
    '''
    y_pred = model.predict(x_test)
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_lb = lb.transform(y_test)
    y_pre_lb = lb.transform(y_pred)
    print(f'accuracy:   {accuracy_score(y_test, y_pred)}')
    print(f"auc score:  {roc_auc_score(y_test_lb, y_pre_lb, average='macro')}")
    print(f"classification report:\n{classification_report(y_test, y_pred, digits=3)}\n")


target_file = 'dataset.tsv'
X_train, X_test, Y_train, Y_test, dataset = pre_process(target_file)
vader_baseline(dataset, Y_test)
random_forest(X_train, X_test, Y_train, Y_test)
mlp(X_train, X_test, Y_train, Y_test)
sgd(X_train, X_test, Y_train, Y_test)

