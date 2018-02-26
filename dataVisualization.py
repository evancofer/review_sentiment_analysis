'''
data_visualization.py
Python Version 2.7

Helper module that generates and outputs various plots & graphs to 
display results of classification of sentiments.
'''
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import sys
import math
import os
import re
import argparse
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from wordcloud import WordCloud
# Local imports.
import preprocessSentences


def pca_viz(data, labels, name):
    """
        Performs PCA and plots to visualize the data.
    """
    pca = PCA()
    prcomp = pca.fit_transform(data)
    plt.figure()
    colors = ['navy', 'darkorange']
    tags = ["Negative", "Positive"]
    lw = 2
    for i in range(1,len(data)):
        lbl = int(labels[i][0])
        plt.scatter(prcomp[i, 0], prcomp[i, 1], color=colors[lbl], alpha=.8, lw=lw,label=tags[lbl])
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(name)
    plt.savefig(name + '.png')

def generate_wordcloud(fname): 
	def pos_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
		return "hsl(150, 175%%, %d%%)" % random.randint(60, 100)

	def neg_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
		return "hsl(0, 175%%, %d%%)" % random.randint(60, 100)
	positives = []
	negatives = []
	
	with open(fname) as f:
		for line in f:
			fields = line.split('\t')
			if fields[2].rstrip('\r\n') == '0':
				negatives.append(fields[1].rstrip('\r\n'))
			else:
				positives.append(fields[1].rstrip('\r\n'))

 	# Generate a word cloud image
 	pos_wordcloud = WordCloud().generate(' '.join(positives))
 	neg_wordcloud = WordCloud().generate(' '.join(negatives))
 	default_colors = pos_wordcloud.to_array()

 	# Display the positive wordcloud:
 	plt.imshow(pos_wordcloud.recolor(color_func=pos_color_func, random_state=3),interpolation="bilinear")
 	plt.axis("off")
 	plt.savefig("pos_wordcloud.png")

 	# Display the negative wordcloud:
 	plt.imshow(neg_wordcloud.recolor(color_func=neg_color_func, random_state=3), interpolation='bilinear')
 	plt.axis("off")
 	plt.savefig("neg_wordcloud.png")

generate_wordcloud(sys.argv[1])