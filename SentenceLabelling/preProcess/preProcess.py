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
utterance_number = 0
current_speaker = ""
first_speaker = True
previous_speaker = ""


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


def replace_username(sentence):
    words = sentence.split()
    result = ""
    for word in words:
        if word.startswith("@"):
            result = result + "username" + " "
        else:
            result = result + word + " "
    return result


def replace_email(sentence):
    words = sentence.split()
    result = ""
    for word in words:
        if word.startswith("www"):
            result = result + "email" + " "
        else:
            result = result + word + " "
    return result


def remove_stop_words(sentence):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(sentence)

    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    result = ""
    for word in filtered_sentence:
        result = result + word + " "

    return result


def replace_numbers(sentence):
    words = sentence.split()
    result = ""
    for word in words:
        if word.isdigit():
            result = result + "number" + " "
        else:
            result = result + word + " "
    return result


def get_pos(sentence):
    pos_tag = ""
    for word, pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
        pos_tag = pos_tag + str(word) + "/" + str(pos) + " "
    return pos_tag


def extract_conversation(train_path, data_path):
    temp_df = pd.DataFrame(columns=['speaker', 'thread_number', 'label', 'type', 'pos', 'utterance'])
    count = 0
    existing_speakers = {}
    speaker_number = 'A'
    speaker_count = 0
    global first_speaker
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

            temp_df.loc[-1] = [to_insert_speaker, num_utterance, row['label'], row['type'], pos_tag, sentence]
            temp_df.index = temp_df.index + 1
    return count


def run_pre_process(train_path, data_path):
    count = extract_conversation(train_path, data_path)
    return count
    