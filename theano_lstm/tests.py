import preprocess
from rnn_numpy import RNNNumpy
import numpy as np

X_train,y_train,vocabulary_size = preprocess.create_train_data()
np.random.seed(10)
model = RNNNumpy(vocabulary_size)
output, hidden_states = model.forward_propagation(X_train[10])
print output.shape
print output

predictions = model.predict(X_train[10])
print predictions.shape
print predictions

# Limit to 1000 examples to save time
print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
print "Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])