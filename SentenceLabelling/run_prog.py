from preProcess import preProcess
import basic_crf
import os
import sys


def main():
    cwd = os.getcwd()
    directory = str(cwd)
    train_path = directory + "\\trainData\\"
    data_path = directory + "\\data\\wellsfargo.csv"
    train_data_percent = 0.9
    number_of_conversations = preProcess.run_pre_process(train_path, data_path)
    basic_crf.run_model(number_of_conversations, train_data_percent)


if __name__ == "__main__": main()

