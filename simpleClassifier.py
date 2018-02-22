'''
simpleClassifier.py
Python Version 2.7

Uses the bag-of-words approach to perform sentiment classification using a few different
ML classification methods. 

Command line arguments:
[0] path to training data file
[1] path to testing data file
'''
import nltk, re, pprint
import sys, math, numpy, scipy
import preprocessSentences
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import svm 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def process_file(path, wordCountThreshold, train):
	if train: 
		(docs, classes, samples, words) = preprocessSentences.tokenize_corpus(path, train)
		vocab = preprocessSentences.wordcount_filter(words, wordCountThreshold)
		return(docs, classes, vocab)
	else:
		(docs, classes, samples) = preprocessSentences.tokenize_corpus(path, train)
		return(docs, classes)  	

def evaluate_results(predictions, testingLabels):
	correct = 0.0
	for i in range(1,len(predictions)):
		if (predictions[i] == testingLabels[i]):
			correct = correct + 1

	print("testing accuracy = ", correct / len(predictions))

def main(argv):
	if (len(argv) != 2):
		raise ValueError("You must specify a training data file and a testing data file")

	(trainingDocs, trainingLabels, trainingVocab) = process_file(argv[0], 4, True)
	trainingData = preprocessSentences.find_wordcounts(trainingDocs, trainingVocab)

	(testingDocs, testingLabels) = process_file(argv[1], 4, False)
	testingData = preprocessSentences.find_wordcounts(testingDocs, trainingVocab)

	svmClf = svm.SVC()
	gnbClf = GaussianNB()
	rfClf = RandomForestClassifier(n_estimators=10)

	svmClf.fit(trainingData, trainingLabels)
	gnbClf.fit(trainingData, trainingLabels)
	rfClf.fit(trainingData, trainingLabels)

	svmPredictions = svmClf.predict(testingData)
	gnbPredictions = gnbClf.predict(testingData)
	rfPredictions = rfClf.predict(testingData)

	print("--------SVM Testing Accuracy------")
	evaluate_results(svmPredictions, testingLabels)
	print("--------Naive Bayes Testing Accuracy------")
	evaluate_results(gnbPredictions, testingLabels)
	print("--------Random Forest Testing Accuracy------")
	evaluate_results(rfPredictions, testingLabels)

if __name__ == "__main__":
  main(sys.argv[1:])