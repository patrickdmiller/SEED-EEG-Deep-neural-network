# Deep CNN for classifying EEG Emotion

you'll need the SEED I dataset. Application here: [ application form](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/)

## Model Design
+ 5 blocks of 2 conv layer with LeakyReLU, maxpool, and residual connections
+ LSTM with residual
+ classification layers