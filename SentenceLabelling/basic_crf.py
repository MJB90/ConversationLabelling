import os

from featureGenerator import features
from featureGenerator import tfidf
from helper_tool import get_data_file_name
import pycrfsuite
import time
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn_crfsuite import CRF

trainer = pycrfsuite.Trainer(verbose=False)
iterations = 100
labels = {'REQ': 0, 'ANSW': 1, 'COMPLIM': 2, 'ANNOU': 3, 'THK': 4, 'RESPOS': 5, 'APOL': 6, 'RCPT': 7, 'COMPLAINT': 8}


def cal_tf_idf(dir_path):
    all_conversations = list(get_data_file_name(dir_path))
    x = []
    for conversation in all_conversations:
        # print(conversation["fileName"])
        utterances = conversation["data"]
        for utterance in utterances:
            x.append(utterance.utterance)

    data = []
    for ob in x:
        data.append(str(ob))
    tfid_vector = tfidf.tf_idf_calculator(data)
    return tfid_vector


def get_conversation_data(dir_path, is_train):
    tfid_vector = cal_tf_idf(dir_path)
    all_conversations = list(get_data_file_name(dir_path))
    x = []
    y = []
    start_index_tfid = 0
    for conversation in all_conversations:
        utterances = conversation["data"]
        x_seq, y_seq = features.get_features(utterances, is_train, tfid_vector, start_index_tfid)
        x.append(x_seq)
        y.append(y_seq)
        start_index_tfid += len(utterances)
    return x, y


def test_for_accuracy(training_dir_path, binary_model_file, test_dir_path):
    global trainer
    global labels

    # Get the training Data
    x_train, y_train = get_conversation_data(training_dir_path, True)

    print("Loaded Training Data")

    for x_sequence, y_sequence in zip(x_train, y_train):
        trainer.append(x_sequence, y_sequence)

    # Get Testing Data
    x_test, y_test = get_conversation_data(test_dir_path, False)

    print("Loaded Testing Data")

    trainer.set_params({
        # coefficient for L1 penalty
        'c1': 0.1,

        # coefficient for L2 penalty
        'c2': 0.001,

        # maximum number of iterations
        'max_iterations': 200,

        # whether to include transitions that
        # are possible, but not observed
        'feature.possible_transitions': True
    })
    # trainer.select(algorithm='lbfgs')
    trainer.train(binary_model_file)
    tagger = pycrfsuite.Tagger()
    tagger.open(binary_model_file)
    y_prediction = [tagger.tag(x_seq) for x_seq in x_test]

    predictions = np.array([labels[tag] for row in y_prediction for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])

    print("CRF :" + str(accuracy_score(truths, predictions)))


def test_accuracy(training_dir_path, test_dir_path):
    global trainer
    global labels

    # Get the training Data
    x_train, y_train = get_conversation_data(training_dir_path, True)
    print("Loaded Training Data")

    for x_sequence, y_sequence in zip(x_train, y_train):
        trainer.append(x_sequence, y_sequence)

    # Get Testing Data
    x_test, y_test = get_conversation_data(test_dir_path, False)
    print("Loaded Testing Data")

    crf = CRF(algorithm='l2sgd',
              c2=0.001,
              max_iterations=100,
              all_possible_transitions=False)

    crf.fit(x_train, y_train)
    y_prediction = crf.predict(x_test)

    predictions = np.array([labels[tag] for row in y_prediction for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])

    print(y_test)
    print(y_prediction)

    print("CRF :" + str(accuracy_score(truths, predictions)))


def train_crf_model(binary_model_file):
    # for param in trainer.get_params():
    #    print(param, trainer.help(param))
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': iterations,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(binary_model_file)

    print("Trained Baseline_CRF model with {} iterations".format(iterations))


def main():
    training_dir_path = "trainData"
    test_dir_path = "testData"
    output_file_name = "result.txt"
    binary_model_file = 'baseline_crfmodel'

    start_time = time.time()
    test_accuracy(training_dir_path, test_dir_path)
    # test_for_accuracy(training_dir_path, binary_model_file, test_dir_path)
    # load_training_data(training_dir_path)
    # train_crf_model(binary_model_file)
    # make_prediction(test_dir_path, binary_model_file, output_file_name)
    print("--- %s seconds ---" % (time.time() - start_time))


# baseline_crf.py
if __name__ == "__main__": main()
