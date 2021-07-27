# Andrew Lorber
# Natural Language Processing
# Project 3 - Text Categorization Using Deep Learning

# Imports
import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# Installs stopwords
import nltk
nltk.download('stopwords')

# Constants
stop_words = set(stopwords.words('english'))
val_split = .2    # Size of validation set (percentage)
words_to_keep = 6000  # For tokenizer
oov_token = "OOV" # For tokenizer
padding = "pre"     # For padding
truncating = "post"  # For padding
max_len = 1800      # For padding

# Gets files from user and transforms them into usable datasets
def process_input_files():
    # Imports files
    # --------------

    # Gets test & train file names
    training_list_file_name = input("Please enter the name of the training file list.\n")
    testing_list_file_name = input("Please enter the name of the testing file list.\n")

    # Creates list of training docs
    training_list_file = open(training_list_file_name, 'r')
    relative_training_path = "/".join(training_list_file_name.split("/")[:-1])  # Relative path of training corpus
    training_doc_list = filter(lambda line: line != "",
                               training_list_file.read().split("\n"))  # List of non-empty lines
    training_doc_list = map(lambda line: relative_training_path + line[1:],
                            training_doc_list)  # Swaps '.' with relative path
    training_doc_list = list(map(lambda line: line.split(" "), training_doc_list))  # Splits line into path & category
    training_list_file.close()

    # Creates list of testing docs
    testing_list_file = open(testing_list_file_name, 'r')
    relative_testing_path = "/".join(testing_list_file_name.split("/")[:-1])  # Relative path of testing corpus
    testing_doc_list = list(
        filter(lambda line: line != "", testing_list_file.read().split("\n")))  # List of non-empty lines
    rel_testing_doc_list = list(
        map(lambda line: relative_testing_path + line[1:], testing_doc_list))  # Swaps '.' with relative path
    testing_list_file.close()

    # Converts lists into sets of word embeddings
    # --------------------------------------------

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []

    #  Converts training list into list of words
    np.random.shuffle(training_doc_list)
    for train_doc, label in training_doc_list:
        y_train.append(label)

        train_file = open(train_doc, 'r')
        train_doc = " ".join(train_file.read().strip().lower().split())

        # Removes stop words
        for stopword in stop_words:
            train_doc = train_doc.replace(f" {stopword.lower()} ", " ")

        x_train.append(train_doc)

        train_file.close()

    #  Converts testing list into list of words
    for test_doc in rel_testing_doc_list:
        test_file = open(test_doc, 'r')
        test_doc = " ".join(test_file.read().strip().lower().split())

        # Removes stop words
        for stopword in stop_words:
            test_doc = test_doc.replace(f" {stopword.lower()} ", " ")

        x_test.append(test_doc)

        test_file.close()

    # Splits training set into training and validation sets
    val_size = int(val_split*len(x_train))
    x_val = x_train[-val_size:]
    x_train = x_train[:-val_size]
    y_val = y_train[-val_size:]
    y_train = y_train[:-val_size]

    # Converts training docs to vectors
    x_tokenizer = keras.preprocessing.text.Tokenizer(num_words=words_to_keep, oov_token=oov_token)
    x_tokenizer.fit_on_texts(x_train)
    x_train = x_tokenizer.texts_to_sequences(x_train)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, 
                    padding=padding, truncating=truncating)
    
    # Converts validation docs to vectors
    x_val = x_tokenizer.texts_to_sequences(x_val)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_len, 
                    padding=padding, truncating=truncating)

    # Converts test docs to vectors
    x_test = x_tokenizer.texts_to_sequences(x_test)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len, 
                    padding=padding, truncating=truncating)
    
    # Converts labels to vectors
    y_tokenizer = keras.preprocessing.text.Tokenizer()
    y_tokenizer.fit_on_texts(y_train + y_val)
    y_train = np.array(y_tokenizer.texts_to_sequences(y_train))
    y_val = np.array(y_tokenizer.texts_to_sequences(y_val))
                                                   # Used for printing to file
    return x_train, y_train, x_val, y_val, x_test, testing_doc_list, y_tokenizer

# Trains the model
def train(x_train, y_train, x_val, y_val):
    epochs = 30

    # Builds model
    model = keras.Sequential([
        Embedding(words_to_keep, 128),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(128)),
        Dropout(.30),
        Dense(256),
        Dropout(.30),
        Dense(256),
        Dropout(.30),
        Dense(6, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x=x_train, y=y_train, batch_size=None, epochs=epochs, validation_data=(x_val, y_val), verbose=2)

    return model

# Tests the model
def predict(model, x_test, y_tokenizer):
    y_test_pred = model.predict(x=x_test)
    y_test_pred = list(map(lambda label: np.argmax(label), y_test_pred))
    
    # Gets file with test labels
    testing_labels_file_name = input("Please enter training file list with labels, or enter 'quit' to just print predictions")

    if(testing_labels_file_name.lower() == "quit"):
        return y_test

    # Creates list of testing labels
    testing_labels_file = open(testing_labels_file_name, 'r')
    y_test_true = filter(lambda line: line != "", 
                            testing_labels_file.read().split("\n"))  # List of non-empty lines
    y_test_true = list(map(lambda line: line.split(" ")[1], 
                            y_test_true))  # Gets category
    y_test_true = np.array(y_tokenizer.texts_to_sequences(y_test_true))
    testing_labels_file.close()

    correct = 0
    for i, label in enumerate(y_test_true):
        if y_test_pred[i] == label:
            correct += 1

    print(f"CORRECT: {correct}")
    print(f"ACCURACY: {correct / len(y_test_true) * 100}%\n")

    return y_test_pred

# Writes predictions to output file
def write_to_output(testing_doc_list, y_test):
# Gets output file name
    out_file_name = input("Please enter the name of the output file.\n")

    # Opens output file
    out_file = open(out_file_name, 'w')

    # Writes predictions to output file
    for (doc_path, prediction) in zip(testing_doc_list, y_test):
        out_file.write(doc_path + " " + prediction.capitalize() + '\n')

    out_file.close()

# Main
x_train, y_train, x_val, y_val, x_test, testing_doc_list, y_tokenizer = process_input_files()

model = train(x_train, y_train, x_val, y_val)

y_test = predict(model, x_test, y_tokenizer)

write_to_output(testing_doc_list, y_tokenizer.sequences_to_texts([y_test])[0].split())
