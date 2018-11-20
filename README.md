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
11/18: Currently, the model was runnable but generating results with low accuracy. Excessive noise within input was impressed. The  next step, I'll try to implement regular expressions and reduce the noice of input. Hopefully, it'll somewhat improve the accuracy of prediction.

11/20 (preprocess_1120.py): I've add the preprocess work using regular expressions. And my naughty network finally learn the pattern of output over 50 iterations! Although the accuracy was still low, but anyway, it's a huge advance for me!
<NOte> After 300 iterations, he displayed significant better performance than before (I thought...).

_Here's the output:_    
source: 10.43; 95%CI 6.90-15.75    
output:  10.00[  0.99 10.75]<pad>    
source: 2.6, 95% CI: 1.8-3.7    
output:    2.6[   1.8   3.7]<pad>    
source: 0.99, 95 CI 0.74-1.32    
output:   0.99[  0.74  1.32]<pad>    
source: 1.09; 95 % CI 0.88 to 1.34    
output:   1.09[  0.88  1.44]<pad>    
source: 0.84, 95% CI 0.43-3.85    
output:   0.44[  0.43  3.55]<pad>    
source: 1.76; 95% CI, 1.45-2.13    
output:   1.76[  1.45  2.11]<pad>    


