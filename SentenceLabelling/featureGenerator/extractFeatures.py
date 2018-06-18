import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score


class ExtractFeature:

    def __init__(self, filename):
        self.filename = filename

    def openfile(self):
        self.data = pd.read_csv(self.filename)
        self.data = pd.DataFrame(self.data)
        self.data['label'].replace('', np.nan, inplace=True)
        self.data.dropna(subset=['label'], inplace=True)
        print(len(self.data))
        self.unique_labels = self.data['label']
        self.unique_labels = np.unique(self.unique_labels)

    def gen_feature(self):
        tf_idf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                 stop_words='english')
        self.features = tf_idf.fit_transform(self.data.utterance).toarray()
        print(self.features.shape)
        print(self.data.utterance)

    def naive_train(self):
        model = GaussianNB()
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.data.label,  test_size=0.33,
                                                            random_state=0)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print("Naive Bayes :" + str(accuracy_score(y_test, y_pred)))

    def svm_train(self):
        self.model = svm.SVC(kernel="rbf", C=100, gamma=0.01)
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.data.label, test_size=0.2,
                                                            random_state=0)
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        print("SVM :" + str(accuracy_score(y_test, y_pred)))









