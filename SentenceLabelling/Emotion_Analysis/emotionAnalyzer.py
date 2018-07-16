import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

utterance_number = 0
current_speaker = ""
first_speaker = True
previous_speaker = ""

sid = SentimentIntensityAnalyzer()


def extract_conversation(train_path, data_path):
    temp_df = pd.DataFrame(columns=['speaker', 'label', 'type', 'neg', 'neu', 'pos', 'utterance'])
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
            ss = sid.polarity_scores(sentence)

            temp_df.loc[-1] = [to_insert_speaker, row['label'], row['type'], ss['neg'],
                               ss['neu'], ss['pos'], sentence]
            temp_df.index = temp_df.index + 1

    return count


def run_pre_process():
    cwd = os.getcwd()
    directory = str(cwd)
    train_path = directory + "\\emotionAnalysisData\\"
    data_path = directory + "\\Edata\\xboxDataId.csv"
    count = extract_conversation(train_path, data_path)
    return count

