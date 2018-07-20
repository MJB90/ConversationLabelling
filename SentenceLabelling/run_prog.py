from preProcess import preProcess
import basic_crf
import os

'''
This is the main function to run the entire analysis code

data_path is the location where where your data file is stored

train_path is the location where the cleaned and pre-processed data needs
to be stored

train_data_percent is the split percentage of test and train data
 
'''


def main():
    cwd = os.getcwd()
    directory = str(cwd)
    train_path = directory + "\\trainData\\"
    data_path = directory + "\\data\\Boost.csv"
    train_data_percent = 0.9

    # Pre Process and get total number of conversations in train data
    number_of_conversations = preProcess.run_pre_process(train_path, data_path)

    # Run the crf model---further training and feature functions are in
    basic_crf.run_model(number_of_conversations, train_data_percent)


if __name__ == "__main__":
    main()

