import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, losses, Model, Input
from tensorflow.keras.layers import Dense, Subtract, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.nn import leaky_relu
from tensorflow.keras.utils import plot_model, Progbar
from sklearn.model_selection import train_test_split

class RN(Model):
  """
  Note: most inspiration for this implementation is from
  https://shorturl.at/UT1FR
  so, I must give credit where credit is due.
  However, I'm trying to improve my DNN skills, so I'm going to go through
  and make notes about what is happening at each part.
  """
  def __init__(self):
    super().__init__()
    # three layer nn, each with a leaky relu (which allows small negative
    # numbers), and a decreasing layer size
    self.steps = [Dense(64, activation=leaky_relu), Dense(16, activation=leaky_relu), Dense(8, activation=leaky_relu)]

    # take the last layer and pass it to an output value
    self.out = Dense(1, activation='linear')

    # used to make the score/probability comparison
    self.diff = Subtract()

  def call(self, pair):
    # get each input row
    i, j = pair

    # run each data point through the first layer
    di = self.steps[0](i)
    dj = self.steps[0](j)

    # continue for all layers, passing through
    for d in range(1, len(self.steps)):
      di = self.steps[d](di)
      dj = self.steps[d](dj)

    # convert to a single value
    output_i = self.out(di)
    output_j = self.out(dj)

    # subtract the two values, then use a sigmoid to plot in [0, 1]
    diff = self.diff([output_i, output_j])
    return Activation('sigmoid')(diff)

def rank_loss(y_true, y_pred):
  correct_greater = tf.cast(y_true == 1.0, tf.float32) * tf.sqrt(tf.maximum(0.0, 1.0 - y_pred))
  correct_less = tf.cast(y_true == 0.0, tf.float32) * tf.sqrt(tf.maximum(0.0, y_pred))

  loss = tf.reduce_mean(correct_greater + correct_less)
  return loss

def train(xi_train, xj_train, labels_train, epochs, batch_size, xi_test, xj_test, labels_test, patience):
  model = RN()

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss=rank_loss)

  early_stopping = EarlyStopping(
      monitor='val_loss',
      patience=patience,
      restore_best_weights=True
  )

  h = model.fit([xi_train, xj_train], labels_train, epochs=epochs, batch_size=batch_size, validation_data=([xi_test, xj_test], labels_test), callbacks=[early_stopping])

  return model

def get_ranking(final, model):
  final_xi = []
  final_xj = []
  final_id = []
  xi_ids = []
  xj_ids = []

  l = 0
  for i in range(len(final)):
    for j in range(i):
      final_xi.append(final[i])
      final_xj.append(final[j])

      xi_ids.append(i)
      xj_ids.append(j)

      final_id.append(l)
      l += 1

  final_xi = np.array(final_xi)
  final_xj = np.array(final_xj)

  predicted_output = model.predict([final_xi, final_xj])
  scores = np.zeros((len(final), len(final)))

  for k in range(l):
    i = xi_ids[k]
    j = xj_ids[k]

    scores[i, j] = predicted_output[k][0]
    scores[j, i] = 1 - predicted_output[k][0]

  ranking = np.sum(scores, axis=1)
  indices = np.argsort(ranking)[::-1]

  return indices

def save_ranking(indices, t, mp):
  results = []
  for ind in indices:
    print(ind)
    results.append(mp['PlayerName'][ind])

    with open(f'../../assets/{t}/rankings.txt', 'w') as file:
      for res in results:
        file.write(f'{res}\n')

    print(results)

def make_pairs(X, y):
  # make pairs
  xi = []
  xj = []
  labels = []
  id = []

  l = 0
  for i in range(len(X)):
    for j in range(i):
      if y[i] == y[j]:
        labels.append(0.5)
      elif y[i] > y[j]:
        labels.append(1)
      else:
        labels.append(0)
      xi.append(X[i])
      xj.append(X[j])

      id.append(l)
      l += 1

  xi = np.array(xi)
  xj = np.array(xj)
  labels = np.array(labels)

  return xi, xj, labels, id