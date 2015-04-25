import re
import numpy as np
from utils import normalize
import math
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk.stem.porter import *
from nltk.collocations import *
import string
import os
import glob
import sys


"""
Author: 
Alex Kong (https://github.com/hitalex)

Reference:
http://blog.tomtung.com/2011/10/plsa
"""
porter = nltk.PorterStemmer()
BIGRAM_WEIGHT = 0.2
TRIGRAM_WEIGHT = 0.3
POS_WEIGHT = 0.4
TOP = 0.1



np.set_printoptions(threshold='nan')

class Document(object):

	'''
	Splits a text file into an ordered list of words.
	'''

	# List of punctuation characters to scrub. Omits, the single apostrophe,
	# which is handled separately so as to retain contractions.
	PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*']

	# Carriage return strings, on *nix and windows.
	CARRIAGE_RETURNS = ['\n', '\r\n']

	# Final sanity-check regex to run on words before they get
	# pushed onto the core words list.
	WORD_REGEX = "^[a-z']+$"


	def __init__(self, filepath):
		'''
		Set source file location, build contractions list, and initialize empty
		lists for lines and words.
		'''
		self.filepath = filepath
		self.file = open(self.filepath)
		self.lines = []
		self.words = []


	def split(self, STOP_WORDS_SET):
		'''
		Split file into an ordered list of words. Scrub out punctuation;
		lowercase everything; preserve contractions; disallow strings that
		include non-letters.
		'''
		self.lines = [line for line in self.file]
		for line in self.lines:
			words = line.split(' ')
			for word in words:
				clean_word = self._clean_word(word)
				if clean_word and (clean_word not in STOP_WORDS_SET) and (len(clean_word) > 1): # omit stop words
					self.words.append(porter.stem(clean_word).encode("ascii"))
		self.file.close()


	def _clean_word(self, word):
		'''
		Parses a space-delimited string from the text and determines whether or
		not it is a valid word. Scrubs punctuation, retains contraction
		apostrophes. If cleaned word passes final regex, returns the word;
		otherwise, returns None.
		'''
		word = word.lower()
		#word = porter.stem(word).encode("ascii")
		for punc in Document.PUNCTUATION + Document.CARRIAGE_RETURNS:
			word = word.replace(punc, '').strip("'")
		return word if re.match(Document.WORD_REGEX, word) else None


class Corpus(object):
	'''
	A collection of documents.
	'''
	def __init__(self):
		'''
		Initialize empty document list.
		'''
		self.documents = []
		#self.term_doc_matrix = np.zeros([number_of_documents, vocabulary_size], dtype = np.int)



	def calculate_lte(self):
		#print "calculate_lte"
		self.term_freqency = np.zeros([self.vocabulary_size],dtype=np.float)
		for w_index in xrange(self.vocabulary_size):
			for d_index in xrange(self.number_of_documents):
				self.term_freqency[w_index] += self.term_doc_matrix[d_index][w_index] 
			self.term_freqency[w_index] = self.term_freqency[w_index] / self.vocabulary_size

		
		for z in xrange(self.number_of_topics):
			for w_index in xrange(self.vocabulary_size):
				self.reverse_topic_term[z][w_index] = self.topic_word_prob[z][w_index] / self.term_freqency[w_index]

		for w_index in xrange(self.vocabulary_size):
			temp = 0
			for z in xrange(self.number_of_topics):
				temp -= self.reverse_topic_term[z][w_index] * math.log(self.reverse_topic_term[z][w_index])
			self.lte_matrix[w_index] = temp

		#print " \n ############# \n \n ######################### \n"
		#print self.lte_matrix
		#print self.vocabulary[7]
		np.savetxt('lte_matrix_word.txt',self.lte_matrix)
		np.savetxt('term_freqency_word.txt',self.term_freqency)
		np.savetxt('term_doc_matrix_doc_word.txt',self.term_doc_matrix)




	def calculate_lts(self): 
		#print "Nirant - calculate_lts"
		self.lts_matrix = np.zeros([self.vocabulary_size, self.number_of_topics], dtype=np.float)
		self.lte_matrix = np.zeros([self.vocabulary_size], dtype=np.float)
		self.reverse_topic_term = np.zeros([self.number_of_topics, self.vocabulary_size],dtype=np.float)
		self.stat_lte = np.zeros([self.vocabulary_size,self.number_of_documents], dtype=np.float)


		for w_index in xrange(self.vocabulary_size):
			for z in xrange(self.number_of_topics):
				num = 0.0
				den = 0.0
				for d_index in xrange(self.number_of_documents):
					num = num + (self.term_doc_matrix[d_index][w_index] * self.document_topic_prob[d_index][z])
					den = den + ( self.term_doc_matrix[d_index][w_index] * (1.0 - self.document_topic_prob[d_index][z]))
				self.lts_matrix[w_index][z] = num/den 
		#print " \n ############# \n \n ######################### \n"
		#print self.lts_matrix
		np.savetxt('lts_matrix_word_topic.txt',self.lts_matrix)


	def calculate_stat_lte(self):
		#print "Nirant calculate_stat_lte"
		lte_scale = 5
		for w_index in xrange(self.vocabulary_size):
			for d_index in xrange(self.number_of_documents):
				self.stat_lte[w_index][d_index] = lte_scale * self.term_doc_matrix[d_index][w_index] / self.lte_matrix[w_index]
		#print self.lte_matrix
		#print " \n ############# \n \n ######################### \n"
		#print self.stat_lte
		#print " \n I am here!!!"
		np.savetxt('stat_lte_word_doc.txt',self.stat_lte)



	def add_document(self, document):
		'''
		Add a document to the corpus.
		'''
		self.documents.append(document)



	def build_vocabulary(self):
		'''
		Construct a list of unique words in the corpus.
		'''
		# ** ADD ** #
		# exclude words that appear in 90%+ of the documents
		# exclude words that are too (in)frequent
		discrete_set = set()
		for document in self.documents:
			for word in document.words:
				#stemmed_word = porter.stem(word).encode("ascii")
				#discrete_set.add(stemmed_word)
				discrete_set.add(word)
		self.vocabulary = list(discrete_set)
		print self.vocabulary
		print len(self.vocabulary)
		


	def plsa(self, number_of_topics, max_iter):
		self.number_of_topics = number_of_topics

		'''
		Model topics.
		'''
		print "EM iteration begins..."
		# Get vocabulary and number of documents.
		self.build_vocabulary = self.build_vocabulary()
		self.number_of_documents = len(self.documents)
		self.vocabulary_size = len(self.vocabulary)
		
		# build term-doc matrix
		self.term_doc_matrix = np.zeros([self.number_of_documents, self.vocabulary_size], dtype = np.int)
		for d_index, doc in enumerate(self.documents):
			term_count = np.zeros(self.vocabulary_size, dtype = np.int)
			#for word in doc(words):
			#stemmed_word = porter.stem(word).encode("ascii")
			for  word in doc.words:
				if word in self.vocabulary:
					w_index = self.vocabulary.index(word)
					term_count[w_index] = term_count[w_index] + 1
			self.term_doc_matrix[d_index] = term_count
		#print self.term_doc_matrix

		# Create the counter arrays.
		self.document_topic_prob = np.zeros([self.number_of_documents, self.number_of_topics], dtype=np.float) # P(z | d) # P[Tk / dj]
		self.topic_word_prob = np.zeros([self.number_of_topics, len(self.vocabulary)], dtype=np.float) # P(w | z) # P[ti / Tk]
		self.topic_prob = np.zeros([self.number_of_documents, len(self.vocabulary), self.number_of_topics], dtype=np.float) # P(z | d, w)

		# Initialize
		print "Initializing..."
		# randomly assign values
		self.document_topic_prob = np.random.random(size = (self.number_of_documents, self.number_of_topics))
		for d_index in range(len(self.documents)):
			normalize(self.document_topic_prob[d_index]) # normalize for each document
		self.topic_word_prob = np.random.random(size = (self.number_of_topics, len(self.vocabulary)))
		for z in range(self.number_of_topics):
			normalize(self.topic_word_prob[z]) # normalize for each topic
		"""  
		# for test, fixed values are assigned, where number_of_documents = 3, vocabulary_size = 15
		self.document_topic_prob = np.array(
		[[ 0.19893833,  0.09744287,  0.12717068,  0.23964181,  0.33680632],
		 [ 0.27681925,  0.22971358,  0.1704416,   0.18248461,  0.14054095],
		 [ 0.24768207,  0.25136754,  0.14392363,  0.14573845,  0.21128831]])

		self.topic_word_prob = np.array(
	  [[ 0.02963563,  0.11659963,  0.06415405,  0.1291839 ,  0.09377842,
		 0.09317023,  0.06140873,  0.023314  ,  0.09486251,  0.01538988,
		 0.09189075,  0.06957687,  0.05015957,  0.05281074,  0.0140651 ],
	   [ 0.09746902,  0.12212085,  0.07635703,  0.02799546,  0.0282282 ,
		 0.03685356,  0.01256655,  0.03931912,  0.09545668,  0.00928434,
		 0.11392475,  0.12089124,  0.02674909,  0.07219077,  0.12059333],
	   [ 0.02209806,  0.05870101,  0.12101806,  0.03733935,  0.02550749,
		 0.09906735,  0.0706651 ,  0.05619682,  0.10672434,  0.12259672,
		 0.04218994,  0.10505831,  0.00315489,  0.03286002,  0.09682255],
	   [ 0.0428768 ,  0.11598272,  0.08636138,  0.10917224,  0.05061344,
		 0.09974595,  0.01647265,  0.06376147,  0.04468468,  0.01986342,
		 0.10286377,  0.0117712 ,  0.08350884,  0.049046  ,  0.10327543],
	   [ 0.02555784,  0.03718368,  0.10109439,  0.02481489,  0.0208068 ,
		 0.03544246,  0.11515259,  0.06506528,  0.12720479,  0.07616499,
		 0.11286584,  0.06550869,  0.0653802 ,  0.0157582 ,  0.11199935]])
		"""
		# Run the EM algorithm
		for iteration in xrange(max_iter):
			print "Iteration #" + str(iteration + 1) + "..."
			print "E step:"
			for d_index, document in enumerate(self.documents):
				for w_index in xrange(self.vocabulary_size):
					prob = self.document_topic_prob[d_index, :] * self.topic_word_prob[:, w_index]
					if sum(prob) == 0.0:
						print "d_index = " + str(d_index) + ",  w_index = " + str(w_index)
						print "self.document_topic_prob[d_index, :] = " + str(self.document_topic_prob[d_index, :])
						print "self.topic_word_prob[:, w_index] = " + str(self.topic_word_prob[:, w_index])
						print "topic_prob[d_index][w_index] = " + str(prob)
						exit(0)
					else:
						normalize(prob)
					self.topic_prob[d_index][w_index] = prob
			print "M step:"
			# update P(w | z)
			for z in xrange(number_of_topics):
				for w_index in xrange(self.vocabulary_size):
					s = 0
					for d_index in xrange(len(self.documents)):
						count = self.term_doc_matrix[d_index][w_index]
						s = s + count * self.topic_prob[d_index, w_index, z]
					self.topic_word_prob[z][w_index] = s
				normalize(self.topic_word_prob[z])
			
			# update P(z | d)
			for d_index in xrange(len(self.documents)):
				for z in range(number_of_topics):
					s = 0
					for w_index in xrange(self.vocabulary_size):
						count = self.term_doc_matrix[d_index][w_index]
						s = s + count * self.topic_prob[d_index, w_index, z]
					self.document_topic_prob[d_index][z] = s
#                print self.document_topic_prob[d_index]
#                assert(sum(self.document_topic_prob[d_index]) != 0)
				normalize(self.document_topic_prob[d_index])

	





