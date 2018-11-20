import numpy as np
import pandas as pd
import random
import keras.backend as K
from keras.utils import to_categorical

def load_data(fname = None):
 human_readable = set()
 machine_readable = set()
 dataset = []
 if fname is not None:
  df = pd.read_csv(fname, encoding = "ISO-8859-1")
 for index, row in df.iterrows():
  dataset.append((row[1].lower(),row[2].lower()))
  human_readable.update(tuple(row[1].lower()))
  machine_readable.update(tuple(row[2].lower()))
  human = dict(zip(sorted(human_readable) + ['<unk>', '<pad>'], list(range(len(human_readable) + 2))))
  inv_machine = dict(enumerate(sorted(machine_readable) + ['<unk>', '<pad>']))
  machine = {v:k for k,v in inv_machine.items()}
 return dataset, human, machine, inv_machine
 
def string_to_int(string, length, vocab):
 string = string.lower()
 string = string.replace(',','')   
 if len(string) > length:
  string = string[:length]       
 rep = list(map(lambda x: vocab.get(x, '<unk>'), string))   
 if len(string) < length:
  rep += [vocab['<pad>']] * (length - len(string))   
 return rep

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
 X, Y = zip(*dataset)
 X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
 Y = [string_to_int(t, Ty, machine_vocab) for t in Y]
 Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
 Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))
 return X, np.array(Y), Xoh, Yoh
 
def softmax(x, axis=1):
 ndim = K.ndim(x)
 if ndim == 2:
  return K.softmax(x)
 elif ndim > 2:
  e = K.exp(x - K.max(x, axis=axis, keepdims=True))
  s = K.sum(e, axis=axis, keepdims=True)
  return e / s
 else:
  raise ValueError('Cannot apply softmax to a tensor that is 1D')
        

