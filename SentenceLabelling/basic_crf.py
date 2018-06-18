import os

import eli5
from featureGenerator import features
from helper_tool import get_data_file_name
import pycrfsuite
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn_crfsuite import CRF

trainer = pycrfsuite.Trainer(verbose=False)
iterations = 100
labels = {'REQ': 0, 'ANSW': 1, 'COMPLIM': 2, 'ANNOU': 3, 'THK': 4, 'RESPOS': 5, 'APOL': 6, 'RCPT': 7,
 'COMPLAINT': 8}


def load_training_data(training_dir_path):
    all_conversations = list(get_data_file_name(training_dir_path))
    for conversation in all_conversations:
        # print(conversation["fileName"])
        utterances = conversation["data"]
        print(utterances)
        x_seq, y_seq = features.get_features(utterances)
        print(x_seq)
        # # print(len(x_seq), len(y_seq))
        trainer.append(x_seq, y_seq)
    print("Loaded Training Data")


def test_for_accuracy(training_dir_path, binary_model_file, test_dir_path):
    global trainer
    global labels

    # Get the training Data
    all_conversations = list(get_data_file_name(training_dir_path))
    x = []
    y = []
    for conversation in all_conversations:
        # print(conversation["fileName"])
        utterances = conversation["data"]
        x_seq, y_seq = features.get_features(utterances)
        x.append(x_seq)
        y.append(y_seq)
    print("Loaded Training Data")

    x_train = x
    y_train = y
    for x_sequence, y_sequence in zip(x_train, y_train):
        trainer.append(x_sequence, y_sequence)

    # Get Testing Data
    all_conversations = list(get_data_file_name(test_dir_path))
    x = []
    y = []
    for conversation in all_conversations:
        # print(conversation["fileName"])
        utterances = conversation["data"]
        x_seq, y_seq = features.get_test_features(utterances)
        x.append(x_seq)
        y.append(y_seq)
    print("Loaded Testing Data")

    x_test = x
    y_test = y
    trainer.set_params({
        # coefficient for L1 penalty
        'c1': 0.1,

        # coefficient for L2 penalty
        'c2': 0.01,

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
    print(y_test)
    print(y_prediction)
    predictions = np.array([labels[tag] for row in y_prediction for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])
    print("CRF :" + str(accuracy_score(truths, predictions)))


def test_accuracy(training_dir_path, test_dir_path):
    global trainer
    global labels

    # Get the training Data
    all_conversations = list(get_data_file_name(training_dir_path))
    x = []
    y = []
    for conversation in all_conversations:
        # print(conversation["fileName"])
        utterances = conversation["data"]
        x_seq, y_seq = features.get_features(utterances)
        x.append(x_seq)
        y.append(y_seq)
    print("Loaded Training Data")

    x_train = x
    y_train = y
    for x_sequence, y_sequence in zip(x_train, y_train):
        trainer.append(x_sequence, y_sequence)

    # Get Testing Data
    all_conversations = list(get_data_file_name(test_dir_path))
    x_ = []
    y_ = []
    for conversation in all_conversations:
        # print(conversation["fileName"])
        utterances = conversation["data"]
        x_seq, y_seq = features.get_test_features(utterances)
        x_.append(x_seq)
        y_.append(y_seq)
    print("Loaded Testing Data")

    x_test = x_
    y_test = y_
    crf = CRF(algorithm='lbfgs',
              c1=0.1,
              c2=0.001,
              max_iterations=100,
              all_possible_transitions=False)

    crf.fit(x_train, y_train)
    y_prediction = crf.predict(x_test)

    predictions = np.array([labels[tag] for row in y_prediction for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])
    print("CRF :" + str(accuracy_score(truths, predictions)))
    eli5.show_weights(crf, top=30)


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


def make_prediction(test_dir_path, binary_model_file, output_file_name):
    tagger = pycrfsuite.Tagger()
    tagger.open(binary_model_file)
    all_conversations = list(get_data_file_name(test_dir_path))
    with open(output_file_name, 'w') as outFile:
        for conversation in all_conversations:
            print("Filename=\"{}\"".format(os.path.basename(conversation["fileName"])), file=outFile)
            utterances = conversation["data"]
            (x_seq, y_seq) = features.get_features(utterances)

            prediction = tagger.tag(x_seq)
            for predicted_tag in prediction:
                print(predicted_tag, file=outFile)
            print("", file=outFile)
    print("Predicted Sequential Labels for Test Data")


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
