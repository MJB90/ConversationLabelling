# ConversationLabelling
Steps:
1.Pre processing 
--change the directory path of train_path (give the path of your trainData folder)
--change the directory path of data_path (give the path of your csv file)

After this step all the conversations goes into trainData Folder along with some of the features.

2.
Copy[20 % or 10% ] whatever files you want to test on to the testData folder from trainData folder.

3.Run basic_crf.py
    In basic_crf.py test_accuracy()[sklearn] and test_for_accuracy()[python crf suite ] are the only used fuctions right now.
