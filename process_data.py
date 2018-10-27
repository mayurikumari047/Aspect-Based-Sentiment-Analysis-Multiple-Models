import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer
import math
source_word2idx = {}
target_word2idx = {}
source_data = []
source_loc_data = []
target_data = []
target_label = []
max_length = 0

def read_data(file):
    data_train_1 = pd.read_csv(file)
    return data_train_1

def read_and_process_data(data_train_1, sourceWord2idx, targetWord2idx):
    global source_word2idx
    global target_word2idx
    source_word2idx = sourceWord2idx
    target_word2idx = targetWord2idx
    #parse_data(data_train_1)
    #create_vocab(data_train_1)
    data_train_1.apply(prepare_data,axis = 1)
    return source_data, source_loc_data, target_data, target_label, max_length


def split_data(data_train_1, train_size, test_size):
    size = data_train_1.shape[0]
    training_rows = math.ceil((train_size/100)*size)
    testing_rows = size - training_rows
    train_data = data_train_1.iloc[0:training_rows]
    test_data = data_train_1.iloc[training_rows:]
    return train_data, test_data


def custom_tokenize(text):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)
    words = [word for word in tokens if word.isalnum()]
    return words


def parse_data(data_train_1):
    data_train_1[' text'] = data_train_1[' text'].apply(lambda x: x.replace('[comma]',',').lower())
    data_train_1[' text'] = data_train_1[' text'].apply(custom_tokenize)
    data_train_1[' aspect_term'] = data_train_1[' aspect_term'].apply(lambda x: x.lower())
    data_train_1[' aspect_term'] = data_train_1[' aspect_term'].apply(custom_tokenize)
    data_train_1[' aspect_term'] = data_train_1[' aspect_term'].apply(lambda x:" ".join(x))
    return data_train_1


def prepare_data(row):
    global max_length
    global source_word2idx
    global target_word2idx
    m = [source_word2idx[id] for id in row[' text']]
    if len(m) == 2602:
        print(row[' text'])
    if len(m) > max_length:
        max_length = len(m)
    source_data.append(m)
    t = [target_word2idx[row[' aspect_term']]]
    target_data.append(t)
    target_label.append(row[' class'])
    get_pos(row)


def get_pos(row):
    index = []
    s_len = len(row[' text']) - 1
    p = row[' text'].copy()
    # print('%s in %s: '%(row[' aspect_term'],row[' text']))
    aspects = row[' aspect_term'].split(' ')
    for aspect in aspects:
        try:
            if len(aspects) - 1 > aspects.index(aspect):
                a_i = [i for i, val in enumerate(row[' text']) if val == aspect]
                try:
                    for a_id in a_i:
                        if row[' text'][a_id + 1] != aspects[aspects.index(aspect) + 1]:
                            a_i.remove(a_id)
                except:
                    pass
                #             index.append(row[' text'].index(aspect))
                index.extend(a_i[0])
            else:
                index.append(row[' text'].index(aspect))
            p[row[' text'].index(aspect)] = s_len
        except:
            pass
    try:
        for i in range(index[0]):
            #             p[i] = index[0] - i
            p[i] = s_len - index[0] + i
        v = s_len
        for i in range(index[len(index) - 1], len(p)):
            # p[i] = i - index[len(index)-1]
            if i == index[len(index) - 1]:
                p[i] = v
            else:
                p[i] = v - 1
                v = v - 1

        # print(p)
    except Exception as e:
        print(e)
        print(p)
        print('exception caught')
        print('%s,%s' % (row[' text'], row[' aspect_term']))
        p = [0 for i in row[' text']]
        print(p)
    source_loc_data.append(p)
    return p


