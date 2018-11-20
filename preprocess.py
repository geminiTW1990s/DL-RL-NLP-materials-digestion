from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.models import load_model, Model
from utils import *

#fname = 'abstract_segs_or_sub300_noallblank.csv'
fname = 'test.csv'
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_data(fname)

Tx = 48 
Ty = 17 
X,Y,Xoh,Yoh=preprocess_data(dataset,human_vocab,machine_vocab,Tx,Ty)

repeator=RepeatVector(Tx)
concatenator=Concatenate(axis=-1)
densor=Dense(1,activation="relu")
activator=Activation(softmax,name='attention_weights')
dotor=Dot(axes=1)

def one_step_attention(a,s_prev):
 s_prev=repeator(s_prev)
 concat=concatenator([a,s_prev])
 e=densor(concat)
 alphas=activator(e)
 context=dotor([alphas,a])
 return context

n_a=64
n_s=128
post_activation_LSTM_cell=LSTM(n_s,return_state=True)
output_layer=Dense(len(machine_vocab),activation=softmax)

def model(Tx,Ty,n_a,n_s,human_vocab_size,machine_vocab_size):
 X=Input(shape=(Tx,human_vocab_size))
 s0=Input(shape=(n_s,),name='s0')
 c0=Input(shape=(n_s,),name='c0')
 s=s0
 c=c0
 outputs=[]   
 a=Bidirectional(LSTM(n_a,return_sequences=True))(X)
 for t in range(Ty):
  context=one_step_attention(a,s)
  s,_,c=post_activation_LSTM_cell(context,initial_state=[s,c])
  out=output_layer(s)
  outputs.append(out)
 model=Model([X,s0,c0],outputs)
 return model

model=model(Tx,Ty,n_a,n_s,len(human_vocab),len(machine_vocab))
opt=Adam(lr=0.005,beta_1=0.9,beta_2=0.999,decay=0.01)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#define all your inputs and outputs to fit the model
m=5280
s0=np.zeros((m,n_s))
c0=np.zeros((m,n_s))
outputs=list(Yoh.swapaxes(0,1))

model.fit([Xoh,s0,c0],outputs,epochs=50,batch_size=100)

EXAMPLES=['R12 (OR = 0.21, 95%CI = 0.07-0.66). Furthermor', 
          'sis (OR 0.53, 95% CI 0.39–0.73), 19 studies, m',
		  'tis (OR = 0.35, 95% CI = 0.17–0.69, N = 3), pr',
		  'ity [OR=0.59, 95% CI (0.39, 0.90), P=0.001] an',
		  'oled OR was 0.40 (95% CI: 0.18-0.86) (I 2 =64%',
		  'eas (OR 1.76, 95 % CI: 1.15 to 2.71) with an o',
		  'ams (OR 1.57; 95% CI, 1.13–2.01; I2 = 76.4%, p',
		  'AEs (OR 1.20 (1.06-1.36) P = 0.005, I2 = 80%) ']
for example in EXAMPLES:
 source=string_to_int(example,Tx,human_vocab)
 source=np.array(list(map(lambda x: to_categorical(x,num_classes=len(human_vocab)),source))).swapaxes(0,1)
 source=source.reshape(55,48,1).swapaxes(0,2)
 prediction=model.predict([source,s0,c0])
 prediction=np.argmax(prediction,axis=-1)
 output=[inv_machine_vocab[int(i)] for i in prediction]
 print("source:",example)
 print("output:",''.join(output))

