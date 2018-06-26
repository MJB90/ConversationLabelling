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


def is_thank_you(utterance):
    if utterance.utterance and (utterance.utterance.find('thank') or utterance.utterance.find('thanks')):
        return 1
    return 0


def get_pos_tokens(utterance):
    tokens = []
    pos_tags = []
    if utterance.pos:
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
        current_act_tag = utterances[utterance_no].label

        tokens, pos_tags = get_pos_tokens(utterance)
        num_punctuation = get_num_punctuation(utterance)
        is_thank = is_thank_you(utterance)
        features = {
            'bias': 1, 'firstUtterance': first_utterance, 'speakerChange': speaker_change,
            'tokens': tokens, 'pos': pos_tags, 'CP': num_punctuation, 'thank': is_thank,
            'tfid': dict
        }

        if istrain:
            if utterance_no > 1:
                prev_utterance = utterances[utterance_no - 1]
                tokens, pos_tags = get_pos_tokens(prev_utterance)
                num_punctuation = get_num_punctuation(prev_utterance)
                is_thank = is_thank_you(prev_utterance)
                features.update({
                    '-1tag': prev_utterance.label, '-1tokens': tokens, '-1pos': pos_tags, '-1CP': num_punctuation,
                    '-1thank': is_thank
                })
            else:
                features['BOS'] = True
        else:
            if utterance_no > 1:
                prev_utterance = utterances[utterance_no - 1]
                tokens, pos_tags = get_pos_tokens(prev_utterance)
                num_punctuation = get_num_punctuation(prev_utterance)
                is_thank = is_thank_you(prev_utterance)
                features.update({
                    '-1tag': "", '-1tokens': tokens, '-1pos': pos_tags, '-1CP': num_punctuation,
                    '-1thank': is_thank
                })
            else:
                features['BOS'] = True

        if istrain:
            if utterance_no < len(utterances) - 1:
                post_utterance = utterances[utterance_no + 1]
                tokens, pos_tags = get_pos_tokens(post_utterance)
                num_punctuation = get_num_punctuation(post_utterance)
                is_thank = is_thank_you(post_utterance)
                features.update({
                    '+1tag': post_utterance.label, '+1tokens': tokens, '+1pos': pos_tags, '+1CP': num_punctuation,
                    '+1thank': is_thank
                })
            else:
                features['EOS'] = True
        else:
            if utterance_no < len(utterances) - 1:
                post_utterance = utterances[utterance_no + 1]
                tokens, pos_tags = get_pos_tokens(post_utterance)
                num_punctuation = get_num_punctuation(post_utterance)
                is_thank = is_thank_you(post_utterance)
                features.update({
                    '+1tag': "", '+1tokens': tokens, '+1pos': pos_tags, '+1CP': num_punctuation,
                    '+1thank': is_thank
                })
            else:
                features['EOS'] = True

        utterance_label = utterance.label

        # Append the tfid vector containing max of 3 grams
        # for key, value in dict.items():
        #     utterance_features.extend([str(key)+"="+str(value)])

        x_seq.append(features)
        y_seq.append(utterance_label)
        utterance_no += 1
        start_index += 1
    return x_seq, y_seq
