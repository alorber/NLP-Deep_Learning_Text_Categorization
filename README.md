###### Natural Language Processing (ECE 467) Project 3
# Deep Learning Text Categorization

Project 3 of Natural Language Processing was to reimplement project 1 (text categorization) using deep learning methods.

The description of project 1 can be read [here](https://github.com/alorber/NLP-Text_Categorization).

I used Tensorflow for this project and my final model was:

Embedding Layer [128] --> Bidirectional LSTM Layer [128] --> Bidirectional LSTM Layer [128] --> Dropout Layer [0.3] --> Dense Layer [256] --> Dropout Layer [0.3] --> Dense Layer [256] --> Dropout Layer [0.3] --> Dense [Output]
