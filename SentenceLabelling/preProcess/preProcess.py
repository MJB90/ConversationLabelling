import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''
->This file is used to extract the various conversations in the file.
->The id of the utterance is replaced by an alphabet(A:Z) depending on the 
  speaker
->The pos of the utterances are extracted and also kept in the output conversation
  files
'''

positive_emoticon = [':-P', ':P', ':-D', ':D', ':p', '=P', '=D', '<3', '>:3', ':-)',
                     ':)', '>-]', '=)', ':o)', ':-*', ':*', ';-)']
negative_emoticon = [":'(", ';*(', '>:[', ':-(', ':(']


# The below global variables are used in find_utterance_number to find the utterance position
# in a particular conversation
utterance_number = 0
current_speaker = ""
first_speaker = True
previous_speaker = ""

'''
This functions replaces the emoticons with posemt and negemt

parameters: utterance of a conversation

return value : the utterance with replaced emoticons
'''


def replace_emoticon(sentence):
    global positive_emoticon
    global negative_emoticon

    words = sentence.split()
    for word in words:
        if word in positive_emoticon:
            sentence = sentence.replace(word, "POSEMT")
        elif word in negative_emoticon:
            sentence = sentence.replace(word, "NEGEMT")
    return sentence


'''
The function below finds the utterance position in the conversation thread
Eg: 
SPEAKER         TEXT                UTTERANCE_POSITION
agent           hi                          1
agent           how are you                 2
customer        I am fine                   1

parameters : current speaker id

return value : position of the utterance
'''


def find_utterance_number(speaker):
    global utterance_number, current_speaker, previous_speaker, first_speaker
    current_speaker = speaker
    if first_speaker:
        first_speaker = False
        utterance_number = 1
        previous_speaker = current_speaker
    elif previous_speaker == current_speaker:
        utterance_number += 1
    elif previous_speaker != current_speaker:
        utterance_number = 1
        previous_speaker = current_speaker

    return utterance_number

# Replacing complex urls with just URL to make it simpler
# and also extra "/"


'''
The function below replaces the words beginning with http as url

parameters : the utterance

return value: the utterance with http.... replaced
'''


def replace_http_url(sentence):
    words = sentence.split()
    result = ""
    for word in words:
        if word.startswith("http"):
            result = result + "url" + " "
        else:
            result = result + word + " "
    result = result.replace("/", "")
    return result


'''
The function below replaces all @_usernames as username

parameter : the utterance

return value : the utterance with @ username replaced 
'''


def replace_username(sentence):
    words = sentence.split()
    result = ""
    for word in words:
        if word.startswith("@"):
            result = result + "username" + " "
        else:
            result = result + word + " "
    return result


'''
The function below replaces all email addresses with email

parameter : the utterance

return value : the utterance with the email address replaced
'''


def replace_email(sentence):
    words = sentence.split()
    result = ""
    for word in words:
        if word.startswith("www"):
            result = result + "email" + " "
        else:
            result = result + word + " "
    return result


'''
This function removes all the stop words from the utterances

parameter : the utterance

return value : the utterance with all the stop words removed
'''


def remove_stop_words(sentence):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(sentence)

    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    result = ""
    for word in filtered_sentence:
        result = result + word + " "

    return result


'''
This function replaces all numeric values with the word number

parameter: the utterance

return value : the utterance with numerical values removed
'''


def replace_numbers(sentence):
    words = sentence.split()
    result = ""
    for word in words:
        if word.isdigit():
            result = result + "number" + " "
        else:
            result = result + word + " "
    return result


'''
The function below finds the parts of speech of each word in the utterance

parameter: the utterance

return value : the parts of speech in the following sentence format
                customer/NN very/ADJ
'''


def get_pos(sentence):
    pos_tag = ""
    for word, pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
        pos_tag = pos_tag + str(word) + "/" + str(pos) + " "
    return pos_tag


'''
This function does the following steps:
--> iterate over each row of the data file

--> apply cleaning steps for each utterance

--> whenever a blank row is found.This marks the end of a conversation
    and store the cleaned data in a csv file and clear the contents of the 
    data frame to store new conversation
    
--> keep counting the number of conversations uploaded 

parameters : train_path --> folder where the conversations are kept
            data_path ---> folder where the main data file is located
            

return value : total number of conversations saved
'''


def extract_conversation(train_path, data_path):
    # Temporary data frame to store the conversation
    temp_df = pd.DataFrame(columns=['speaker', 'thread_number', 'label', 'type', 'pos', 'utterance'])

    # variable to count total number of conversations
    count = 0

    # Map to name different speakers
    existing_speakers = {}
    # speaker name starts with A which is followed by other alphabets
    speaker_number = 'A'
    # variable to count the total number of speakers in a conversation
    speaker_count = 0
    global first_speaker

    # Read the data file
    df = pd.read_csv(data_path, skip_blank_lines=False)

    for index, row in df.iterrows():
        if pd.isnull(row['utterance']) and len(temp_df) > 1:

            # update file count
            count = count + 1
            filename = train_path + str(count) + ".csv"
            temp_df.to_csv(filename, index=False)

            # Clear the temporary data frame and dictionary for making it
            # ready for the next conversation data frame
            temp_df = temp_df.iloc[0:0]
            existing_speakers.clear()
            speaker_count = 0

            # update the first_speaker as a new conversation will begin in the
            # next iteration
            first_speaker = True
            print("Conversation " + str(count) + "is uploaded!")

        elif pd.notnull(row['utterance']) and pd.notnull(row['label']):
            # Different Speakers are assigned a different alphabet
            curr_speaker = str(row['speaker'])

            if curr_speaker in existing_speakers:
                to_insert_speaker = existing_speakers[curr_speaker]

            else:
                existing_speakers[curr_speaker] = chr(ord(speaker_number) + speaker_count)
                to_insert_speaker = existing_speakers[curr_speaker]
                speaker_count = speaker_count + 1

            sentence = row['utterance']

            # Convert to lower case
            sentence = sentence.lower()

            # Replace http... with url
            sentence = replace_http_url(sentence)

            # Replace www.... with email
            sentence = replace_email(sentence)

            # Replace numbers with string numbers
            sentence = replace_numbers(sentence)

            # Replace @u_name with username
            sentence = replace_username(sentence)

            # Replace emoticon with pos or neg emotion
            sentence = replace_emoticon(sentence)

            # Remove stop words from the utterance
            sentence = remove_stop_words(sentence)

            # Extract the pos of each token
            pos_tag = get_pos(sentence)

            # Get the conversation thread number
            num_utterance = find_utterance_number(to_insert_speaker)

            # Keep adding processed to temporary data frame
            # This data belongs to a single conversation
            temp_df.loc[-1] = [to_insert_speaker, num_utterance, row['label'], row['type'], pos_tag, sentence]
            temp_df.index = temp_df.index + 1

    return count


'''
The function below runs the entire pre-processing algorithm and returns the
total number of conversations
'''


def run_pre_process(train_path, data_path):
    count = extract_conversation(train_path, data_path)
    return count
    