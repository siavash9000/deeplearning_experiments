import preprocess
from rnn_numpy import RNNNumpy
from rnn_theano import RNNTheano
import numpy as np
import cProfile

X_train,y_train,vocabulary_size = preprocess.create_train_data()
np.random.seed(10)
model = RNNNumpy(vocabulary_size)

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
#cProfile.run("model.numpy_sdg_step(X_train[10], y_train[10], 0.005)")
#print("----------------------------------------------------------------")
np.random.seed(10)
model_theano = RNNTheano(vocabulary_size)
#cProfile.run("model_theano.train_with_sgd(X_train[10], y_train[10], 0.005)")
print("----------------------------------------------------------------")
losses_numpy = model.train_with_sgd(X_train[:100], y_train[:100], nepoch=5, evaluate_loss_after=1)
losses_theano = model_theano.train_with_sgd(X_train[:100], y_train[:100], nepoch=5, evaluate_loss_after=1)