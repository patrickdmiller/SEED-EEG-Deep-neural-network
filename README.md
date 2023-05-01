# Deep CNN for classifying EEG Emotion

you'll need the SEED I dataset. Application here: [ application form](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/)

## Pytorch model design
+ 5 blocks of 2 conv layer with LeakyReLU, maxpool, dropout, and residual connections
+ LSTM with residual
+ classification layers with dropout, ReLU
+ softmax, ce

## Use
+ extract SEED I data to ./data
+ ```./CNN-LSTM-RES.sh```