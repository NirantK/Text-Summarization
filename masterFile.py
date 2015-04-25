import json
import os
import glob
import sys
from main import main

def extractText(datapath):
	'''Iterate over entire folder'''
	exceptions = []
	for document_path in datapath:
		for document_file in glob.glob(os.path.join(document_path, '*.txt')):
			with open(document_file, "r+") as doc:
				try:
					new_filename = document_path+"clean/"+os.path.basename(document_file).split('.')[0] + "_clean.txt"
					with open(new_filename, "w+") as f:
						j = json.load(doc)
						for i in xrange(len(j)):
							f.write(j[str(i)]["caption"]["content"].encode('utf-8'))
							f.write(" ")
				except Exception as e:
					# print 'Removing...',doc, e
					# os.remove(os.path.join(document_file))
					# print 'Exception: ', e
					exceptions.append(e)
	return exceptions

def printList(l):
	for x in l:
		print x

if __name__=="__main__": 
	'''
	1. Extract text from JSON Files
	2. Score these text files using all the functions
	3. Score these text files without PLSA scores
	4. Do Corpus Reduction for both of these using PLSA Sentence Similarity Score
	5. Do corpus reduction Han's SSS
	6. Plot Comparative Graphs
	'''
	datapath = ['./demo_dataset/']
	errors = extractText(datapath)
	print 'The following files were missing from the sequence:',printList(errors)
	print 
	main(datapath[0]+"clean/")
	