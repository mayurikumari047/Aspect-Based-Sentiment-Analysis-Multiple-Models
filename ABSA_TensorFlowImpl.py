from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import *
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, r2_score

import tensorflow as tf
import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer
import math


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

def init_word_embeddings(word2idx):

    import numpy as np
    import codecs
    wt = np.random.normal(0, 0.03, [len(word2idx), 300])
    f = codecs.open("data/glove.6B.300d.txt", "r", "utf-8")
    for line in f:
        content = line.strip().split()
        if content[0] in word2idx:
            wt[word2idx[content[0]]] = np.array(content[1:])
    return wt


def create_vocab(data_train_1):
    source_word2idx = {'<pad>': 0}
    target_word2idx = {}
    for words in data_train_1[' text']:
        for word in words:
            if word not in source_word2idx:
                source_word2idx[word] = len(source_word2idx)

    for words in data_train_1[' aspect_term']:
        if words not in target_word2idx:
            target_word2idx[words] = len(target_word2idx)
    return source_word2idx, target_word2idx



def absa_nn_tensor_flow(train_vectors, test_vectors, train_labels, test_labels):
    
    learning_rate = 0.05
    epochs = 100
    batch_size = 100
    example_size = train_vectors.shape[1]
    print("example_size:", example_size)

    # declare the training data placeholders
    x = tf.placeholder(tf.float32, [None, example_size])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 3])

    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random_normal([example_size, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([300]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random_normal([300, 3], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([3]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)

    test_labels_list = list()
    for k in range(len(test_labels)):
        if test_labels[k] == -1:
            test_lbl = [1.0, 0.0, 0.0]
        elif test_labels[k] == 0:
            test_lbl = [0.0, 1.0, 0.0]
        elif test_labels[k] == 1:
            test_lbl = [0.0, 0.0, 1.0]
        test_labels_list.append(test_lbl)
    test_labels_final = np.array(test_labels_list)
    print("test_labels_final shape:", test_labels_final.shape)

    merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter('C:\\Mayuri_local\\DMTM\\SA1')
    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        # total_batch = int(len(train_labels) / batch_size)
        total_batch = int(math.floor(len(train_vectors) / batch_size))
        for epoch in range(epochs):
            avg_cost = 0
            i = 0
            for _ in range(total_batch):
                j = 1
                batch_x_list = list()
                batch_y_list = list()

                while (j <= 100):
                    batch_x_list.append(train_vectors[i])

                    if train_labels[i] == -1:
                        target_y = [1.0, 0.0, 0.0]
                    elif train_labels[i] == 0:
                        target_y = [0.0, 1.0, 0.0]
                    elif train_labels[i] == 1:
                        target_y = [0.0, 0.0, 1.0]
                    batch_y_list.append(target_y)
                    j = j + 1
                batch_x = np.array(batch_x_list)
                batch_y = np.array(batch_y_list)
                #print("batch_y shape:", batch_y.shape)
                i = i + 1

                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch

            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
            summary = sess.run(merged, feed_dict={x: test_vectors, y: test_labels_final})
            #writer.add_summary(summary, epoch)

        print("\nTraining complete!")
        #writer.add_graph(sess.graph)
        print(sess.run(accuracy, feed_dict={x: test_vectors, y: test_labels_final}))


if __name__ == "__main__":
    
    source_word2idx = {}
    target_word2idx = {}
    source_data = []
    source_loc_data = []
    target_data = []
    target_label = []
    max_length = 0
    
    train_file = 'data/data_1_train.csv'
    data = read_data(train_file)
    parsed_data = parse_data(data)
    source_word2idx, target_word2idx = create_vocab(parsed_data)
    
    training_data, testing_data = split_data(parsed_data, 80, 20)
    train_data = read_and_process_data(training_data, source_word2idx, target_word2idx)
    test_data = read_and_process_data(testing_data, source_word2idx, target_word2idx)
    
    sen_max_len = train_data[4]
    train_example_list = list()
    for i in range(len(train_data[1])):
        x1 = list()
    
        x1.extend(train_data[1][i])
        if len(x1) < sen_max_len:
            for _ in range(sen_max_len - len(x1)):
                x1.append(0)
        train_example_list.append(x1)
    
    final_array = np.array(train_example_list)
    print("final_array.shape: ",final_array.shape)
    
    labels = train_data[3]
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(final_array, labels, test_size=0.3, random_state=0)

    absa_nn_tensor_flow(train_vectors, test_vectors, train_labels, test_labels)

