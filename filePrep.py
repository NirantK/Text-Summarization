import json
import os
import glob
import sys

datapath = ['./dataset/']

'''Iterate over entire folder'''
for document_path in datapath:
	for document_file in glob.glob(os.path.join(document_path, '*.txt')):
		with open(document_file, "r+") as doc:
			try:
				new_filename = "./dataset/clean/"+os.path.basename(document_file).split('.')[0] + "_clean.txt"
				with open(new_filename, "w+") as f:
					j = json.load(doc)
					for i in xrange(len(j)):
						f.write(j[str(i)]["caption"]["content"].encode('utf-8'))
						f.write(" ")
			except Exception as e:
				# print 'Removing...',doc, e
				# os.remove(os.path.join(document_file))
				print 'Mofo, you got error: ', e