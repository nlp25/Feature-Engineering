#------------------------------------------------------------------------------
# text normalization
#------------------------------------------------------------------------------

import re
import unicodedata

def normalize(text):
    # change to lowercase
    text = text.lower()

    # map specific words before normalizing
    text = re.sub(r"\b(?:é)\b", "eh", text)  # to avoid mixing "é" (verb) with "e" (conjunction)

    # map all diacritic characters to ASCII
    # (Unicode Normalization Form KD (NFKD): Compatibility Decomposition)
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")

    # separate relevant punctualtion by spaces, so it will be taken as a word
    text = re.sub("[ ]*[:()?!][ ]*", lambda match: " " + match.group().strip() + " ", text)

    # remove repeated vowels (used informally in PT-BR text to indicate prolonged vocalization)
    text = re.sub("[a]+|[e]+|[i]+|[o]+|[u]", lambda match: match.group()[0], text)

    # keep only the interesting characters (remove all others)
    text = re.sub(r"[^a-z:()?!\s]", "", text)

    # reduce sequences of space charaters to a single space
    text = re.sub(r"\s+", " ", text)

    return text

#------------------------------------------------------------------------------
# data loading
#------------------------------------------------------------------------------

import pandas as pd

def text_classification_dataset(csv_filepath, normalized=True):
    dataset = pd.read_csv(csv_filepath)

    first_col = dataset.iloc[:,  0]  # first column
    last_col  = dataset.iloc[:, -1]  # last column

    text_samples, int_labels = [], []
    for n, text in enumerate(first_col):
        if type(text) is str:
            text_samples += [normalize(text) if normalized else text]
            int_labels += [int(last_col[n])]

    return text_samples, int_labels

#------------------------------------------------------------------------------
# feature extraction
#------------------------------------------------------------------------------

import collections

def word_frequency(text_samples, min_count=1):
    full_text = " ".join([text for text in text_samples if type(text) is str]) \
                if type(text_samples) in (list, tuple, set) else text_samples  \
                if type(text_samples) is str else ""

    # split the text into a list of words (as separated by spaces in the text)
    all_words = full_text.split()

    # count each word occurence in the list
    counter = collections.Counter(all_words)

    # list of pairs, each one containing a word and its count, reverse sorted by the count (largest first)
    word_counts = sorted([(word, counter[word]) for word in counter if counter[word] >= min_count],
                  key=lambda item: item[1], reverse=True)

    return dict(word_counts)

#------------------------------------------------------------------------------

def fingerprint_reference_words(text_samples):
    return word_frequency(text_samples, min_count=3)

#------------------------------------------------------------------------------

def fingerprint(text, reference_words):
    counts = dict(word_frequency(text))

    result = []
    for word in reference_words:
        result += [counts[word]] if word in counts else [0]

    return result

#------------------------------------------------------------------------------
# model training
#------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense

def create_model(input_size, layer_sizes):
    model = Sequential()  # feedforward

    model.add(Input(shape=(input_size,)))
    # fully connected
    for n, layer_size in enumerate(layer_sizes[0:-1]):
        model.add(Dense(units=layer_size, activation='relu', name="hidden_" + str(n+1)))
    model.add(Dense(units=1, activation='tanh', name="last"))

    return model

#------------------------------------------------------------------------------

def train_model(layer_sizes, text_samples, int_labels, fingerprint_words, verbose=True):
    fingerprints = [fingerprint(text, fingerprint_words) for text in text_samples]

    input_size = len(fingerprints[0])
    model = create_model(input_size, layer_sizes)

    model.compile(loss='mean_squared_error', optimizer='sgd')
    # model.compile(loss='mean_squared_error', optimizer='RMSProp')

    model.fit(fingerprints, int_labels, epochs=10, batch_size=1, verbose=verbose)
    # model.fit(fingerprints, labels, epochs=5, batch_size=4, verbose=True)

    return model

#------------------------------------------------------------------------------
# model evaluation
#------------------------------------------------------------------------------

import numpy as np

def classification_errors(model, text_samples, int_labels, fingerprint_words):
    fingerprints = [fingerprint(text, fingerprint_words) for text in text_samples]

    predicted_values = model.predict(fingerprints)

    # round predicted values (to become predicted labels)
    if len(set(int_labels)) == 2:
        # in case of binary labels, map predicted values to the interval [0, 1] before rounding (then map back)
        predicted_labels = np.round((predicted_values - min_label) / (max_label - min_label))
        predicted_labels = predicted_labels * (max_label - min_label) + min_label
    else:
        predicted_labels = np.round(predicted_values)

    return [{"pos": n, "label": label, "prediction": float(predicted_values[n]), "text": text_samples[n]}
            for n, label in enumerate(int_labels) if label != predicted_labels[n]]

#------------------------------------------------------------------------------

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

def kfold_validation(layer_sizes, text_samples, int_labels, k=5, stratified=True, verbose=True):
    if stratified:
        skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=True)
        folds = skf.split(text_samples, int_labels)
    else:
        kf = KFold(n_splits=k, random_state=None, shuffle=True)
        folds = kf.split(text_samples)

    if verbose:
        print(("Stratified " if stratified else "") + "K-Fold Cross-Validation", "(K="+str(k)+"):")

    min_accuracy, max_accuracy, final_accuracy = 1, 0, 0

    for n, (train_indexes, test_indexes) in enumerate(folds):

        train_text_samples = [text_samples[i] for i in train_indexes]
        train_int_labels = [int_labels[i] for i in train_indexes]

        fingerprint_words = fingerprint_reference_words(train_text_samples)

        model = train_model(layer_sizes, train_text_samples, train_int_labels, fingerprint_words, verbose=verbose)

        test_text_samples = [text_samples[i] for i in test_indexes]
        test_int_labels = [int_labels[i] for i in test_indexes]

        error_cases = classification_errors(model, test_text_samples, test_int_labels, fingerprint_words)
        accuracy = 1 - len(error_cases) / len(int_labels)

        final_accuracy += accuracy
        min_accuracy = min(accuracy, min_accuracy)
        max_accuracy = max(accuracy, max_accuracy)

        if verbose:
            print("Accuracy", "(Fold "+str(n+1)+"/"+str(k)+"):", "{:.2%}".format(accuracy))

    final_accuracy /= k

    if verbose:
        print("Final Accuracy:", "{:.2%}".format(final_accuracy),
              "(min: {:.2%},".format(min_accuracy), "max: {:.2%}".format(max_accuracy)+")")

    return final_accuracy

###############################################################################

#------------------------------------------------------------------------------
# data preparation and model cross-validation
#------------------------------------------------------------------------------

text_samples, int_labels = text_classification_dataset("dataset/tips_scenario1_train.csv", normalized=False)
# text_samples, int_labels = text_classification_dataset("dataset/tips_scenario2_train.csv", normalized=True)

kfold_validation([800, 400, 200, 1], text_samples, int_labels, k=5, stratified=True, verbose=True)
# kfold_validation([400, 400, 1], text_samples, int_labels, k=5, stratified=True, verbose=True)

#------------------------------------------------------------------------------
# model training over full dataset
#------------------------------------------------------------------------------

fingerprint_words = fingerprint_reference_words(text_samples)

model = train_model([800, 400, 200, 1], text_samples, int_labels, fingerprint_words, verbose=True)
# model = train_model([400, 400, 1], text_samples, int_labels, fingerprint_words, verbose=True)
# model.summary()

error_cases = classification_errors(model, text_samples, int_labels, fingerprint_words)
accuracy = 1 - len(error_cases) / len(int_labels)
print("{:.2%}".format(accuracy))

model.predict([fingerprint(normalize("Tudo muito ótimo!"), fingerprint_words)])
model.predict([fingerprint(normalize("Até que tava legal..."), fingerprint_words)])
model.predict([fingerprint(normalize("Até que tava ok..."), fingerprint_words)])
model.predict([fingerprint(normalize("Não gostei muito."), fingerprint_words)])
model.predict([fingerprint(normalize("Tudo muito ruim, péssimo!"), fingerprint_words)])

#------------------------------------------------------------------------------
