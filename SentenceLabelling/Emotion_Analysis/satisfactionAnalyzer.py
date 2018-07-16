from helper import get_data_file_name
import matplotlib.pyplot as plt
import os
import pandas as pd

conversation_number = 0
file_number = 1


def get_agent(utterances):
    customer = ""
    for utterance in utterances:
        if utterance.utterance.find("@XboxSupport2"):
            customer = utterance.speaker
    return customer


def get_request_sentiment(utterance_number, customer_utterances):
    continuous_request = 0
    if utterance_number > 0:
        while utterance_number >= 0 and customer_utterances[utterance_number].label == "REQ":
            continuous_request += 1
            utterance_number -= 1

    if continuous_request == 1:
        return 0

    continuous_request = continuous_request * (-0.5)
    return continuous_request


def plot_graph(points):
    plt.plot(points)
    plt.suptitle("Conversation :" + str(conversation_number))
    plt.xlabel("utterance number")
    plt.ylabel("Sentiment")
    plt.show()


def sentiments_customer(customer_utterances):
    temp_df = pd.DataFrame(columns=['utterance_number', 'score'])
    global file_number
    neg_customer = 0
    pos_customer = 0
    no_utterance_customer = 0
    points = []
    for utterance in customer_utterances:
        neg_customer -= float(utterance.neg)
        pos_customer += float(utterance.pos)
        request_sentiment = get_request_sentiment(no_utterance_customer, customer_utterances)
        neg_customer -= float(request_sentiment)
        curr = neg_customer + pos_customer
        curr = curr * 100
        curr = int(curr)
        points.append(curr)
        no_utterance_customer += 1
        temp_df.loc[-1] = [no_utterance_customer, curr]
        temp_df.index = temp_df.index + 1

    plot_graph(points)

    file_name = "result\\" + str(file_number) + ".csv"
    file_number += 1
    temp_df.to_csv(file_name, index=False)

    print("Number of customer utterance :" + str(no_utterance_customer))
    print(str(neg_customer) + "   AGENT   " + str(pos_customer))


def sentiments_agent(utterances):
    neg_agent = 0
    pos_agent = 0
    customer = get_agent(utterances)
    customer_utterances = []
    for utterance in utterances:
        if utterance.speaker == customer:
            customer_utterances.append(utterance)
        else:
            neg_agent -= float(utterance.neg)
            pos_agent += float(utterance.pos)

    sentiments_customer(customer_utterances)
    customer_utterances.clear()
    print("Length :" + str(len(utterances)))
    print(str(neg_agent) + "   AGENT   " + str(pos_agent))
    print("------------------------------------------------------------------------")
    # plot_graph(points)
    if conversation_number > 10:
        exit(1)


def get_conversation_data(dir_path):
    global conversation_number
    all_conversations = list(get_data_file_name(dir_path))
    for conversation in all_conversations:
        conversation_number += 1
        utterances = conversation["data"]
        sentiments_agent(utterances)


cwd = os.getcwd()
directory = str(cwd)
train_path = directory + "\\emotionAnalysisData\\"
get_conversation_data(train_path)