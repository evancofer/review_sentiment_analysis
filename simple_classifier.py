'''
simpleClassifier.py
Python Version 2.7

Uses the bag-of-words approach to perform sentiment classification using a few different
ML classification methods.

Command line arguments:
[0] path to training data file
[1] path to testing data file
'''
from __future__ import print_function
import sys
import math
import os
import re
import argparse
import numpy as np
import scipy as sp
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
# Local imports.
import preprocessSentences


# Command line arguments.
parser = argparse.ArgumentParser(description="Sentiment Analysis")
parser.add_argument('--verbose', dest='verbose', action="store_true")
parser.add_argument('--train', dest='train_file', default="train.txt")
parser.add_argument('--test', dest='test_file', default="test.txt")
parser.add_argument('--run-test', dest='test', action="store_true")
parser.add_argument('--val-frac', dest="validation_frac", default=0.20, type=float)
parser.add_argument('--val-type', dest="validation_type", default="single")
parser.add_argument('--no-shuffle', dest="shuffle", action="store_false")
args = parser.parse_args()

def process_file(path, wc_threshold, train):
    if train:
        (docs, classes, samples, words) = preprocessSentences.tokenize_corpus(path, train)
        vocab = preprocessSentences.wordcount_filter(words, wc_threshold)
        classes = np.asarray(classes).reshape((len(classes), 1))
        return(docs, classes, vocab)
    else:
        (docs, classes, samples) = preprocessSentences.tokenize_corpus(path, train)
        classes = np.asarray(classes).reshape((len(classes), 1))
        return(docs, classes, None)


def evaluate_results(pred_lbls, true_lbls):
    """
        Evaluate performance by comparing predictions to ground truth.
        Returns a dictionary of various result metric.
    """
    ret = {}
    correct = 0
    for i in range(0, len(pred_lbls)):
        if pred_lbls[i] == true_lbls[i]:
            correct += 1
    ret["accuracy"] = float(correct) / len(pred_lbls)
    return ret


def main(train_fname, test_fname, val_frac, val_type,
        shuffle=True, test=False, verbose=False):
    """
        Main program logic.
        train_fname: Path to training data.
        test_fname: Path to testing data.
        val_frac: Fraction of training data to use for validation.
        val_type: Either "cross" or "single" for now.
        shuffle: If true, shuffle the training data before validation split.
        test: If true, evaluate the test data.
        verbose: Print everything.
    """
    # Build models.
    def build_models():
        models = {"svm": svm.SVC(),
                  "gnb": GaussianNB(),
                  "rf": RandomForestClassifier(n_estimators=10)}
        for model_name in models.keys():
            models[model_name] = {"model": models[model_name]}
            models[model_name].update({"train": {}, "validation": {}})
            if test:
                models[model_name].update({"test": {}})
        return models

    def predict_data(model_dict, name, dataset, data, lbls):
        """
            Uses model to labels for data and evaluates the results.
        """
        model_dict[name][dataset]["predictions"] = model_dict[name]["model"].predict(data)
        model_dict[name][dataset]["performance"] = evaluate_results(model_dict[name][dataset]["predictions"], lbls)
        return model_dict

    def print_performance(model_dict, name, dataset):
        """
            Prints performance for model.
        """
        if verbose is True:
            print("%s performance for: \"%s\"" % (dataset.title(), name))
            for metric_name, metric_value in iter(sorted(model_dict[name][dataset]["performance"].iteritems())):
                print("\t%s:%s" % (metric_name, metric_value))
        return model_dict

    # Get training data and shuffle it.
    (train_docs, train_lbls, train_vocab) = process_file(train_fname, 4, True)
    train_data = preprocessSentences.find_wordcounts(train_docs, train_vocab)
    if shuffle:
        perm = list(np.random.permutation(train_data.shape[0]))
        train_data = train_data[perm, :]
        train_lbls = train_lbls[perm, :]
        train_docs = [train_docs[i] for i in perm]

    if test:
        datasets = ["test", "validation"]
    else:
        datasets = ["validation"]
    # Split data into folds.
    folds = []
    if val_type == "single":
        n = np.ceil(val_frac*train_data.shape[0])
        folds = [(0, np.ceil(val_frac*train_data.shape[0]))]
    elif val_type == "cross":
        n_folds = int(max(1, np.ceil(1. / val_frac)))
        step = train_data.shape[0] / float(n_folds)
        for i in range(int(n_folds)):
            fold_start = int(max(0, i*step))
            fold_end = int(min(train_data.shape[0], step*(i+1)))
            folds.append((fold_start, fold_end))
    if test:
        (test_docs, test_lbls, test_vocab) = process_file(test_fname, 4, False)
        test_data = preprocessSentences.find_wordcounts(test_docs, train_vocab)
        test_lbls = np.asarray(test_lbls)

    global_models = build_models()
    for i, (fold_start, fold_end) in enumerate(folds):
        models = build_models()
        cur_val_data = train_data[fold_start:fold_end,]
        cur_val_lbls = train_lbls[fold_start:fold_end,].flatten()
        cur_val_docs = train_docs[fold_start:fold_end]
        cur_train_data = np.vstack([train_data[fold_start:,], train_data[:fold_end,]])
        cur_train_lbls = np.vstack([train_lbls[fold_start:,], train_lbls[:fold_end,]]).flatten()
        cur_train_docs = sum(train_docs[fold_start:], []) + sum(train_docs[:fold_end], [])

        #TODO: Keep information on the training performance (e.g. time)
        # Training, validation, and testing.
        for model_name in models.keys():
            models[model_name]["model"].fit(cur_train_data, cur_train_lbls)
        for model_name in models.keys():
            models = predict_data(models, model_name, "validation", cur_val_data, cur_val_lbls)
        if test:
            for model_name in models.keys():
                models = predict_data(models, model_name, "test", test_data, test_lbls)

        for dataset in datasets:
            for model_name in models.keys():
                #models[model_name].pop("model")
                if "performance" not in global_models[model_name][dataset]:
                    global_models[model_name][dataset].update({"performance": {}})
                for metric_name in models[model_name][dataset]["performance"]:
                    if metric_name in global_models[model_name][dataset]["performance"]:
                        global_models[model_name][dataset]["performance"][metric_name] += (models[model_name][dataset]["performance"][metric_name] / float(n_folds))
                    else:
                        global_models[model_name][dataset]["performance"][metric_name] = (models[model_name][dataset]["performance"][metric_name] / float(n_folds))
    for model_name in global_models.keys():
        for dataset in datasets:
            print_performance(global_models, model_name, dataset)

def test_input_path(fname):
    """
        Ensures that a given input file exists.
    """
    if not os.path.exists(fname):
        raise ValueError("Could not find file at \"%s\"" % fname)


if __name__ == "__main__":
    for i_fname in [args.train_file, args.test_file]:
        test_input_path(i_fname)
    main(args.train_file, args.test_file, args.validation_frac, args.validation_type,
         args.shuffle, args.test, args.verbose)
