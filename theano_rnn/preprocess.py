import nltk, itertools
import numpy as np

max_vocabulary_size = 9000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
input_file = '../data/emma.txt'

def create_train_data():
    #print "tokenizing raw data"
    data = open(input_file, 'r').read()
    sentences = itertools.chain(*[nltk.sent_tokenize(data.decode('utf-8').lower())])
    sentences = ["%s %s %s" % (sentence_start_token, line, sentence_end_token) for line in sentences]
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    #print "Found %d unique words tokens." % len(word_freq.items())

    vocab = word_freq.most_common(max_vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    #print "Using vocabulary size %d." % len(vocab)
    #print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])


    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


    # Print an training data example
    #x_example, y_example = X_train[17], y_train[17]
    #print "------------------------------------------------------------------------------------------------------------"
    #print "Examples:"
    #print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example)
    #print "\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example)
    #print "------------------------------------------------------------------------------------------------------------"

    vocabulary_size = len(vocab)
    return X_train,y_train, vocabulary_size, index_to_word, word_to_index