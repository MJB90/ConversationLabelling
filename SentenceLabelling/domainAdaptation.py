from preProcess import preProcessForDomain
import basic_crf
import os
import sys


def main():
    cwd = os.getcwd()
    directory = str(cwd)
    test_path = directory + "\\testData\\"
    data_path = directory + "\\data\\wellsfargo.csv"
    preProcessForDomain.run_pre_process(test_path, data_path, True)

    train_path = directory + "\\trainData\\"
    data_path = directory + "\\data\\Boost.csv"

    preProcessForDomain.run_pre_process(train_path, data_path, False)

    basic_crf.run_model(0, 0)


if __name__ == "__main__": main()

