import preprocess
from rnn_numpy import RNNNumpy
from rnn_theano import RNNTheano
import numpy as np
import cProfile

UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"

X_train,y_train,vocabulary_size,index_to_word, word_to_index = preprocess.create_train_data()
np.random.seed(10)
model = RNNNumpy(vocabulary_size)

np.random.seed(10)
model = RNNNumpy(vocabulary_size)

def generate_sentence(model, index_to_word, word_to_index, min_length=5):
    # We start the sentence with the start token
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(new_sentence)[-1]
        samples = np.random.multinomial(1, [next_word_probs])
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        # Seomtimes we get stuck if the sentence becomes too long, e.g. "........" :(
        # And: We don't want sentences with UNKNOWN_TOKEN's
        if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return None
    if len(new_sentence) < min_length:
        return None
    return new_sentence


#cProfile.run("model.numpy_sdg_step(X_train[10], y_train[10], 0.005)")
#print("----------------------------------------------------------------")
np.random.seed(10)
model_theano = RNNTheano(vocabulary_size)
#cProfile.run("model_theano.train_with_sgd(X_train[10], y_train[10], 0.005)")
print("----------------------------------------------------------------")
#losses_numpy = model.train_with_sgd(X_train[:100], y_train[:100], nepoch=5, evaluate_loss_after=1)
losses_theano = model_theano.train_with_sgd(X_train[:100], y_train[:100], nepoch=5, evaluate_loss_after=1)
generated_sentence = generate_sentence(model,index_to_word,word_to_index)
print generated_sentence