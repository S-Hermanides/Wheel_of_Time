import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def display_topics(model, feature_names, no_top_words, topic_names=None):
    """
    Takes an NLP topic model output and displays the specified number of words
    with the highest weights for each topic
    """
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '", topic_names[ix], "'")
        print(", ".join([feature_names[i]
                         for i in topic.argsort()[:-no_top_words - 1:-1]]))


def find_unique(string):
    """
    Find number of unique words in a string
    """
    string_set = set(string.split(' '))
    return len(string_set)


def get_main_topics(df, column_names):
    """
    Return main topic for each document (row)
    """
    topics = []
    sub_array = df[column_names].to_numpy()
    for row in sub_array:
        idx = np.argsort(row)[-1]
        topics.append(column_names[idx])
    return topics


def get_sentiment(text, score='compound'):
    """
    Get Vader Sentiment scores for each document (row)
    """
    sia = SentimentIntensityAnalyzer()
    sent_dict = sia.polarity_scores(text)
    return sent_dict[score]
