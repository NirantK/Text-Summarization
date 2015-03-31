import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk.stem.porter import *
from nltk.collocations import *
import string
import os
import glob
import sys
import io

TOP = 0.1
stopwords = sw.words('english')

def trigramScore(text):
	trigram_measures = nltk.collocations.TrigramAssocMeasures()
	# tokens = nltk.wordpunct_tokenize(text.translate(None, string.punctuation))
	tokens = nltk.wordpunct_tokenize(text)
	tri_finder = TrigramCollocationFinder.from_words(tokens)
	tri_finder.apply_word_filter(lambda w: w in stopwords)
	tri_scored = tri_finder.score_ngrams(trigram_measures.pmi)
	score, val = 0, 0
	for key, scores in tri_scored:
		score += scores
	if len(tokens)!=0:
		val = score/len(tokens)
	return val

def bigramScore(text):
	'''Input:	text
	Output:		information score on the basis of bigram_score'''
	bigram_measures = nltk.collocations.BigramAssocMeasures()
	tokens = nltk.wordpunct_tokenize(text)	  
	# tokens = nltk.wordpunct_tokenize(text.translate(None, string.punctuation))	 
	bi_finder = BigramCollocationFinder.from_words(tokens)
	bi_finder.apply_word_filter(lambda w: w in stopwords)
	bi_scored = bi_finder.score_ngrams(bigram_measures.student_t)
	score, val = 0, 0
	for key, scores in bi_scored:
		score += scores
	if len(tokens)!=0:
		val = score/len(tokens)
	return val

def sym(sentence, s2):
	sentence = [i for i in sentence.split() if i not in stopwords]
	for i in sentence:
		i = stemmer.stem(i)
	print sentence
	for word in sentence:
		completeSet = wn.synsets(word)
		for i in xrange(len(completeSet)):
			print completeSet[i], completeSet[i].wup_similarity(completeSet[i-1])

def partOfSpeechScore(text):
	# Used when tokenizing words
	sentence_re = r'''(?x)      # set flag to allow verbose regexps
		  ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
		| \w+(-\w+)*            # words with optional internal hyphens
		| \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
		| \.\.\.                # ellipsis
		| [][.,;"'?():-_`]      # these are separate tokens
	'''
	lemmatizer = nltk.WordNetLemmatizer()
	stemmer = nltk.stem.porter.PorterStemmer()
	 
	'''Grammar taken from S. N. Kim, T. Baldwin, and M.-Y. Kan. Evaluating n-gram based evaluation metrics for automatic keyphrase extraction. Technical report, University of Melbourne, Melbourne 2010.''' 
	grammar = r'''
		NBAR:
			{<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
			
		NP:
			{<NBAR>}
			{<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
	'''
	chunker = nltk.RegexpParser(grammar)
	 
	toks = nltk.regexp_tokenize(text, sentence_re)
	postoks = nltk.tag.pos_tag(toks)	 
	tree = chunker.parse(postoks)

	def leaves(tree):
		"""Finds NP (nounphrase) leaf nodes of a chunk tree."""
		for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
		# for subtree in tree.subtrees():
			yield subtree.leaves()

	def normalise(word):
		"""Normalises words to lowercase and stems and lemmatizes it."""
		word = word.lower()
		word = lemmatizer.lemmatize(word)
		# word = stemmer.stem_word(word)
		return word

	def acceptable_word(word):
		"""Checks conditions for acceptable word: length, stopword."""
		accepted = bool(2 <= len(word) <= 40
			and word.lower() not in stopwords)
		return accepted
	
	def get_terms(tree):
		for leaf in leaves(tree):
			term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
			yield term

	def printTerms(terms):
		for term in terms:
			for word in term:
				print word,len(term),"\t"
			print
	
	def scoreTerms(terms):
		score = 0
		for term in terms:
			score += len(term)
		return score
	score = scoreTerms(get_terms(tree))
	return float(score)/len(toks)

def documentDriver():
	BIGRAM_WEIGHT = 0.2
	TRIGRAM_WEIGHT = 0.3
	POS_WEIGHT = 0.4
	datapath = ['./dataset/clean/test']
	for document_path in datapath:
		for document_file in glob.glob(os.path.join(document_path, '*.txt')):
			mockingJay = []
			f = io.open(document_file, "r+", encoding='utf-8')
			try:
				print "Reading file ... ",os.path.basename(document_file)
				document = f.read()
				f.close()
				textlines = nltk.sent_tokenize(document)
				print "Calculating sentence score..."
				for line in textlines:
					line = line.translate("\n")
					# mockingJay.append([line, bigramScore(line)+ trigramScore(line)+ partOfSpeechScore(line), bigramScore(line), trigramScore(line), partOfSpeechScore(line)])
					mockingJay.append([line, BIGRAM_WEIGHT*bigramScore(line)+ TRIGRAM_WEIGHT*trigramScore(line)+ POS_WEIGHT*partOfSpeechScore(line)])	
				l = int(len(mockingJay)*TOP)
				print ("Extracting the top %0.2f percent sentences for summarization." %(TOP*100))
				mockingJay = sorted(mockingJay, key = lambda x: x[1], reverse = True)[:l]
				new_filename = "./dataset/summaries/"+os.path.basename(document_file).split('.')[0] + "Summaries.txt"
				print "Writing to file...", os.path.basename(new_filename)
				s = ""
				with open(new_filename, "w+") as f:
					for ele in mockingJay:
						s += ele[0].encode('utf-8') + "\n"
					f.write(s)
				print ("-------------------------------------------------------------")
			except Exception as e:
				# print 'Removing...',doc, e
				# os.remove(os.path.join(document_file))
				print '@NK, you got error: ', e

documentDriver()