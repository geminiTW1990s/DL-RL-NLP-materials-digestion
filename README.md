# Simple notes to record the digestion of DL/RL/NLP materials
## (Warning: as a newcomer, mistaken understanding is unavoidable. -> keep modifying and updating !)

## Neural network (respiratory: NN-transplant)
#### network geometry
#### gradient compute - activating function
#### weight update
#### loss function

## Convolution neural network
#### Basically, abstraction from layer to layer with filters.
#### In NLP, it possesses several computational advantages over combining RNNs.
###### [keyword] attention: like weights for the encoded words in a sentence, which tells us which word is more important for translation
###### [keyword] encoder-decoder design: intuitive and well-implemented with multiple RNNs, connecting words with their encoded hidden states
#### One applications: text classification

## RL - Deep Q network
#### Basically, CNNs with input - states and outputs - q function (kind of function to measure reward conditioning on state-action pair)
#### Not yet found real-world application for NLP

## RL in NLP
#### Reference: Lecture 16 from "CS224n: Natural Language Processing with Deep Learning" of Stanford University


# Implementations
## Successfully run Coursera example
#### NMT_dateformat.py and pre-trained the weights nmt_dateformat_50ep_weights.h5

## Applying LSTMs and attention networks for information extraction from Pubmed abstracts
#### preprocess.py - the network part implementation
#### utils.py - storing functions mostly used during data-preprocessing
#### Trouble shooting - 
&nbsq;&nbsq;&nbsq;&nbsq;Currently, the model was runnable but generating results with low accuracy. Excessive noise within input was impressed. The  next step, I'll try to implement regular expressions and reduce the noice of input. Hopefully, it'll somewhat improve the accuracy of prediction.
