
import sys
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np

def process_preds_max(pred):
	[nr, nc] = pred.shape
	newpreds = np.zeros((nr,nc))
	for rr in range(0,nr):
		maxIdx=np.where(pred[rr,:]==max(pred[rr,:]))[0]
		newpreds[rr,maxIdx]=1
	return newpreds

def assign_topics(gamma, gammamax, tfs):
	numTFs = len(tfs)

	for ii in range(0,numTFs):
		ss = gamma[(ii*200):((ii+1)*200),:]  # blocks of 200 oligomers in gamma file from inference on selex
		ss = (ss.sum(axis=0))/200
		ssmax = gammamax[(ii*200):((ii+1)*200),:]
		ssmax = (ssmax.sum(axis=0))/200
		for topic in range(0,len(ss)):
			print('Topic:%d TF: %s Score: %g Perc-score: %g' % (topic,tfs[ii],ss[topic],ssmax[topic]))
	return

	# get prediction on Tn5 "probes"
	ss = gamma[(numTFs*200):gamma.shape[0],:]
	ss = ss.sum(axis=0)/(gamma.shape[0]-(numTFs*200))
	for topic in range(0,len(ss)):
		print('Topic:%d TF: Tn5 Score: %g' % (topic,ss[topic]))

if (__name__ == '__main__'):

	if (len(sys.argv) != 4):
		print('usage: python evalPerf.py <gamma-file> <oligo-TF-order-file>\n')
		print('supplied arguments: ',len(sys.argv))
		sys.exit(1)

	gamma_file = sys.argv[1]
	selex_tfs_file = sys.argv[2]

	tfnames = open(selex_tfs_file, 'r').readlines()
	tfnames = list(map(lambda x: x.strip(), tfnames))
	gamma=np.load(gamma_file)
	print(gamma.shape)

	ntopics = gamma.shape[1]

	gammamax = process_preds_max(gamma)
	assign_topics(gamma,gammamax,tfnames)
