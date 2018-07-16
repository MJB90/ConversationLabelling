import glob
import os
from featureGenerator import features
from featureGenerator import featureDictionary
from featureGenerator import tfidf
from helper_tool import get_data_file_name
import time
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn_crfsuite import CRF
import pandas as pd
from sklearn.metrics import classification_report

labels = {'REQ': 0, 'ANSW': 1, 'COMPLIM': 2, 'ANNOU': 3, 'THK': 4, 'RESPOS': 5, 'APOL': 6, 'RCPT': 7, 'COMPLAINT': 8,
          'GREET': 9, 'SOLVED': 10, 'OTH': 11}

type_labels = {'OP': 0, 'SV': 1, 'CL': 2, 'CC': 3}


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


def get_conversation_data(dir_path, is_train, is_Convo_label):
    tfid_vector = cal_tf_idf(dir_path)
    all_conversations = list(get_data_file_name(dir_path))
    x = []
    y = []
    start_index_tfid = 0
    for conversation in all_conversations:
        utterances = conversation["data"]
        x_seq, y_seq = featureDictionary.get_features(utterances, is_train, tfid_vector, start_index_tfid, is_Convo_label)
        x.append(x_seq)
        y.append(y_seq)
        start_index_tfid += len(utterances)
    return x, y


def test_accuracy(training_dir_path, test_dir_path, is_Convo_label):
    global labels, type_labels
    curr_labels = {}
    if is_Convo_label:
        curr_labels = labels
    else:
        curr_labels = type_labels

    # Get the training Data
    x_train, y_train = get_conversation_data(training_dir_path, True, is_Convo_label)
    print("Loaded Training Data")

    # Get Testing Data
    x_test, y_test = get_conversation_data(test_dir_path, False, is_Convo_label)
    print("Loaded Testing Data")

    crf = CRF(algorithm='l2sgd',
              c2=0.001,
              max_iterations=100,
              all_possible_transitions=False)

    crf.fit(x_train, y_train)

    y_prediction = crf.predict(x_test)

    predictions = np.array([curr_labels[tag] for row in y_prediction for tag in row])
    truths = np.array([curr_labels[tag] for row in y_test for tag in row])

    # Print Metrics
    if is_Convo_label:
        print(classification_report(
            truths, predictions,
            target_names=['REQ', 'ANSW', 'COMPLIM', 'ANNOU', 'THK', 'RESPOS', 'APOL', 'RCPT']))

    # Get test accuracy
    test_ = str(accuracy_score(truths, predictions))
    # for w in sorted(crf.transition_features_, key=crf.transition_features_.get, reverse=True):
    #     print(str(w) + ":" + str(crf.transition_features_[w]))

    # Testing on training data without label
    x_test, y_test = get_conversation_data(training_dir_path, False, is_Convo_label)
    y_prediction = crf.predict(x_test)

    predictions = np.array([curr_labels[tag] for row in y_prediction for tag in row])
    truths = np.array([curr_labels[tag] for row in y_test for tag in row])

    sf = crf.state_features_
    print(type(sf))
    # Get train accuracy
    train_ = str(accuracy_score(truths, predictions))
    return test_, train_


def test_train_split(number_of_conversations, train_data_percent, training_dir_path, test_dir_path):
    num_in_train = number_of_conversations * train_data_percent
    num_in_test = int(number_of_conversations - num_in_train)
    dialog_filenames = sorted(glob.glob(os.path.join(training_dir_path, "*.csv")))
    count = 0
    cwd = os.getcwd()
    directory = str(cwd)
    for dialog_filename in dialog_filenames:
        df = pd.read_csv(dialog_filename)
        dialog_filename = dialog_filename.replace("trainData\\", "")
        to_remove = directory + "\\trainData" + "\\" + str(dialog_filename)
        os.remove(to_remove)
        file_name = test_dir_path + str(dialog_filename)
        df.to_csv(file_name, index=False)
        count += 1
        if count == num_in_test:
            break


def run_model(number_of_conversations, train_data_percent):
    training_dir_path = "trainData"
    test_dir_path = "testData\\"

    start_time = time.time()
    test_train_split(number_of_conversations, train_data_percent, training_dir_path, test_dir_path)
    test_convo, train_convo = test_accuracy(training_dir_path, test_dir_path, True)

    test_, train_ = test_accuracy(training_dir_path, test_dir_path, False)

    print("-----Train and Test accuracy for conversation labels------")
    print("CRF train accuracy :" + str(train_convo))
    print("CRF test accuracy :" + str(test_convo))

    print()

    print("-----Train and Test accuracy for type labels------")
    print("CRF train accuracy :" + str(train_))
    print("CRF test accuracy :" + str(test_))

    print("--- %s seconds ---" % (time.time() - start_time))
