import re

'''
Feature format:
result = [
is_first_utterance,
is_speaker_change,
previous_tag,
previous-previous_tag,
tokens,
pos,
num_of_punctuation,
tf_id_vector
]
'''


def get_pos_tokens(utterance):
    tokens = []
    pos_tags = []
    for token_pos in utterance.pos:
        tokens.append('TOKEN_' + token_pos.token)
        pos_tags.append('POS_' + token_pos.pos)
    return tokens, pos_tags


def get_num_punctuation(utterance):
    result = 0
    if utterance.utterance:
        result += utterance.utterance.count('#')
        result += utterance.utterance.count('?')
        result += utterance.utterance.count('!')
        result += utterance.utterance.count(',')
    return result


# Generating features for one conversation at a time i.e utterances belongs to a single conversation


def get_features(utterances, istrain, tfid_vector, start_index_tfid_vector):
    x_seq = []
    y_seq = []
    utterance_no = 0
    start_index = start_index_tfid_vector
    for utterance in utterances:
        dict = tfid_vector[start_index]
        first_utterance = False
        speaker_change = False
        if utterance_no == 0:
            first_utterance = True
            current_speaker = utterances[utterance_no].speaker
        else:
            previous_utterance_no = utterance_no - 1
            previous_speaker = utterances[previous_utterance_no].speaker
            current_speaker = utterances[utterance_no].speaker
            if previous_speaker != current_speaker:
                speaker_change = True
                previous_act_tag = utterances[previous_utterance_no].label
                current_act_tag = utterances[utterance_no].label
        # print("{} {} {}".format(first_utterance, speaker_change, current_speaker))
        current_act_tag = utterances[utterance_no].label

        utterance_features = []

        # Check whether first utterance or speaker change taking place
        utterance_features.extend([str(int(first_utterance)), str(int(speaker_change))])

        # Find previous and previous-previous tag
        if istrain:
            if utterance_no > 1:
                previous_act_tag = utterances[utterance_no - 1].label
                previous_to_previous_act_tag = utterances[utterance_no - 2].label
                utterance_features.extend(["PREV_TO_PREV_" + previous_to_previous_act_tag, "PREV_" + previous_act_tag])

        else:
            if utterance_no > 1:
                previous_act_tag = utterances[utterance_no - 1].label
                previous_to_previous_act_tag = utterances[utterance_no - 2].label
                utterance_features.extend(["PREV_TO_PREV_" + "", "PREV_" + ""])

        utterance_label = utterance.label

        # Append the tokens and their respective pos
        if utterance.pos:
            tokens, pos_tags = get_pos_tokens(utterance)
            utterance_features.extend(tokens)
            utterance_features.extend(pos_tags)

        # Append the number of punctuation symbols
        num_punctuation = get_num_punctuation(utterance)
        utterance_features.extend(["CP="+str(num_punctuation)])

        # Append the position of conversation in the thread
        utterance_features.extend(["thread_number="+str(utterance.thread_number)])

        # Append the tfid vector containing max of 3 grams
        # for key, value in dict.items():
        #     utterance_features.extend([str(key)+"="+str(value)])

        # Append isThank you if sentence contains thank or thanks
        if utterance.utterance and (utterance.utterance.find('thank') or utterance.utterance.find('thanks')):
            utterance_features.extend(["thank="+"1"])
        else:
            utterance_features.extend(["thank="+"0"])

        # Append the utterance features to the feature list of training data
        x_seq.append(utterance_features)
        y_seq.append(utterance_label)
        utterance_no += 1
        start_index += 1
    return x_seq, y_seq
