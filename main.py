import os
import glob
import sys
from operator import itemgetter # for sort
import plsa
import math
import io
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk.stem.porter import *
from nltk.collocations import *
import string


porter = nltk.PorterStemmer()
BIGRAM_WEIGHT = 0.2
TRIGRAM_WEIGHT = 0.3
POS_WEIGHT = 0.4
TOP = 0.1
STAT_WEIGHT = 0.4
STOP_WORDS_SET = set()
PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*']
CARRIAGE_RETURNS = ['\n', '\r\n']
WORD_REGEX = "^[a-z']+$"

def print_topic_word_distribution(corpus, number_of_topics, topk, filepath):
	"""
	Print topic-word distribution to file and list @topk most probable words for each topic
	"""
	V = len(corpus.vocabulary) # size of vocabulary
	assert(topk < V)
	f = open(filepath, "w")
	for k in range(number_of_topics):
		word_prob = corpus.topic_word_prob[k, ] # word probability given a topic
		# print word_prob
		word_index_prob = []
		for i in range(V):
			word_index_prob.append([i,corpus.vocabulary[i],word_prob[i]])
		word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True) # sort by word count
		with open('word_index_prob.txt',"a+") as f2:
			f2.write(str(word_index_prob)+'\n')
			f2.close()
		f.write("Topic #" + str(k) + ":\n")
		for i in range(topk):
			index = word_index_prob[i][0]
			f.write(corpus.vocabulary[index] + " ")
		f.write("\n")
	print "Written topic-word distribution to file: " + filepath        
	f.close()
	
def print_document_topic_distribution(corpus, number_of_topics, topk, filepath):
	"""
	Print document-topic distribution to file and list @topk most probable topics for each document
	"""
	# print topk, number_of_topics
	assert(topk < number_of_topics)
	f = open(filepath, "w")
	D = len(corpus.documents) # number of documents
	for d in range(D):
		topic_prob = corpus.document_topic_prob[d, ] # topic probability given a document
		topic_index_prob = []
		for i in range(number_of_topics):
			topic_index_prob.append([i, topic_prob[i]])
		topic_index_prob = sorted(topic_index_prob, key=itemgetter(1), reverse=True)
		f2 = open('topic_index_prob.txt',"a+")
		f2.write(str(topic_index_prob)+'\n')
		f2.close()
		f.write("Document #" + str(d) + ":\n")
		for i in range(topk):
			index = topic_index_prob[i][0]
			f.write("topic" + str(index) + " ")
		f.write("\n")
	print "Written document-topic distribution to file: " + filepath       
	f.close()

def scoreSentence(corpus):
	STAT_WEIGHT = 0.4
	BIGRAM_WEIGHT = 0.2
	TRIGRAM_WEIGHT = 0.3
	POS_WEIGHT = 0.4
	datapath = ['./texts/txt']
	doc_id = 0
	line_number = 1
	for document_path in datapath:
		for document_file in glob.glob(os.path.join(document_path, '*.txt')):
			sentenceList = []
			f = open(document_file, "r")
			print "Reading file ... ",os.path.basename(document_file)
			print "Calculating sentence score..."
			line_number = 1
			for line in f:
				bScore = BIGRAM_WEIGHT*bigramScore(line)
				tScore = TRIGRAM_WEIGHT*trigramScore(line)
				sScore = STAT_WEIGHT*statScore(line,doc_id,corpus)
				sentenceList.append([line, bScore + tScore + sScore, line_number])
				line_number = line_number+ 1 
			l = int(len(sentenceList)*TOP)
			print ("Extracting the top %0.2f percent sentences for summarization." %(TOP*100))
			sentenceList = sorted(sentenceList, key = lambda x: x[1], reverse = True)[:l]
			sentenceList = sorted(sentenceList, key = lambda x: x[2], reverse = False)[:l]            
			new_filename = "./dataset/summaries/set2"+os.path.basename(document_file).split('.')[0] + "Summaries.txt"
			print "Writing to file...", os.path.basename(new_filename)
			s = ""
			with open(new_filename, "w+") as f2:
				for ele in sentenceList:
					s += ele[0].encode('utf-8') + "\n"
				f2.write(s)
				f2.close()
			print ("-------------------------------------------------------------")
			f.close()

			doc_id += 1

def main(argv):
	try:
		document_topk = int(argv[1])
		topic_topk = int(argv[2])
		number_of_topics = int(argv[3])
		max_iterations = int(argv[4])

		if document_topk > number_of_topics:
			raise Exception
	except:
		print "Usage: python ./main.py <document_topk> <topic_topk> <number_of_topics> <maxiteration> "
		print "Necessary condition: document_topk < number_of_topics"
		sys.exit(0)

	# load stop words list from file
	stopwordsfile = open("stopwords.txt", "r")
	for word in stopwordsfile: # a stop word in each line
		word = word.replace("\n", '')
		word = word.replace("\r\n", '')
		STOP_WORDS_SET.add(word)
	stopwordsfile.close()
	
	corpus = plsa.Corpus() # instantiate corpus
	# iterate over the files in the directory.
	document_paths =['./texts/txt']
	#document_paths = ['./test/']
	for document_path in document_paths:
		for document_file in glob.glob(os.path.join(document_path, '*.txt')):
			document = plsa.Document(document_file) # instantiate document
			document.split(STOP_WORDS_SET) # tokenize
			corpus.add_document(document) # push onto corpus documents list

	corpus.build_vocabulary()
	print "Vocabulary size:" + str(len(corpus.vocabulary))
	print "Number of documents:" + str(len(corpus.documents))

	corpus.plsa(number_of_topics, max_iterations)

	# My Code from here! [Saurabh]
	corpus.calculate_lts()
	corpus.calculate_lte()
	corpus.calculate_stat_lte()
	datapath = ['./texts/txt']
	doc_id = 0
	line_number = 1
	for document_path in datapath:
		for document_file in glob.glob(os.path.join(document_path, '*.txt')):
			sentenceList = []
			f = open(document_file, "r")
			print "Reading file ... ",os.path.basename(document_file)
			print "Calculating sentence score..."
			line_number = 1
			for line in f:
				bScore = BIGRAM_WEIGHT*bigramScore(line)
				tScore = TRIGRAM_WEIGHT*trigramScore(line)
				sScore = STAT_WEIGHT*statScore(line,doc_id,corpus)
				sentenceList.append([line, bScore + tScore + sScore, line_number])
				line_number = line_number+ 1 
			l = int(len(sentenceList)*TOP)
			print ("Extracting the top %0.2f percent sentences for summarization." %(TOP*100))
			sentenceList = sorted(sentenceList, key = lambda x: x[1], reverse = True)[:l]
			sentenceList = sorted(sentenceList, key = lambda x: x[2], reverse = False)[:l]            
			new_filename = "./dataset/summaries/set2"+os.path.basename(document_file).split('.')[0] + "Summaries.txt"
			print "Writing to file...", os.path.basename(new_filename)
			s = ""
			with open(new_filename, "w+") as f2:
				for ele in sentenceList:
					s += ele[0].encode('utf-8') + "\n"
				f2.write(s)
				f2.close()
			print ("-------------------------------------------------------------")
			f.close()
			doc_id += 1

#pass a document into similarity_sentences to be re-ordered
#corpus.documents(i) will be sent
def similarity_sentences(doc,number_of_topics,corpus):
	number_of_sentences = len(doc.lines)
	topic_sentence_matrix = np.zeros([number_of_topics,number_of_sentences])
	stemmed_lines = []
	for line in doc.lines:
		stemmed_line = []
		for term in line:
			clean_word = clean_word_func(term)
			if clean_word and (clean_word not in STOP_WORDS_SET) and (len(clean_word) > 1):
				stemmed_word = porter.stem(clean_word).encode("ascii")
				stemmed_line.append(stemmed_word)
		stemmed_lines.append(stemmed_line)

	num = 0.0
	den = 0.0
	line_number = 0
	for line in stemmed_lines:
		for topic_k in xrange(number_of_topics):
			for term in line:				
				w_index = corpus.vocabulary.index(term)
				num = num + corpus.topic_word_prob[topic_k][w_index]
				den = den + 1
			topic_sentence_matrix[topic_k][line_number] = num / den
			num = 0.0
			den = 0.0
		line_number += 1


	sentence_similarity_matrix = np.zeros([number_of_sentences,number_of_sentences])
	for sentence_i in xrange(number_of_sentences):
		for sentence_j in xrange(number_of_sentences):
			if sentence_i == sentence_j:
				continue
			for topic_k in xrange(number_of_topics):
				for term in stemmed_lines[sentence_j]:
					w_index = corpus.vocabulary.index(term)
					sentence_similarity_matrix[sentence_i][sentence_j] += corpus.lts_matrix[w_index][topic_k] * topic_sentence_matrix[topic_k][sentence_i]

#  Normalize it!!!

def statScore(text,d_index,corpus):
	words = text.split(' ')
	val = 0
	for word in words:
		clean_word = clean_word_func(word)
		if clean_word and (clean_word not in STOP_WORDS_SET) and (len(clean_word) > 1): # omit stop words
			stemmed_word = porter.stem(clean_word).encode("ascii")
			w_index = corpus.vocabulary.index(stemmed_word)
			val = val + corpus.stat_lte[w_index][d_index]
	# print val, d_index , " Stat Score\n"
	return val

def clean_word_func(word):
	'''
	Parses a space-delimited string from the text and determines whether or
	not it is a valid word. Scrubs punctuation, retains contraction
	apostrophes. If cleaned word passes final regex, returns the word;
	otherwise, returns None.
	'''
	word = word.lower()
	#word = porter.stem(word).encode("ascii")
	for punc in PUNCTUATION + CARRIAGE_RETURNS:
		word = word.replace(punc, '').strip("'")
	return word if re.match(WORD_REGEX, word) else None





def trigramScore(text):
	trigram_measures = nltk.collocations.TrigramAssocMeasures()
	# tokens = nltk.wordpunct_tokenize(text.translate(None, string.punctuation))
	tokens = nltk.wordpunct_tokenize(text)
	tri_finder = TrigramCollocationFinder.from_words(tokens)
	tri_finder.apply_word_filter(lambda w: w in STOP_WORDS_SET)
	tri_scored = tri_finder.score_ngrams(trigram_measures.pmi)
	score, val = 0, 0
	for key, scores in tri_scored:
		score += scores
	if len(tokens)!=0:
		val = score/len(tokens)
	# print val,  " Trigram Score\n"
	return val

def bigramScore(text):
	'''Input:   text
	Output:     information score on the basis of bigram_score'''
	bigram_measures = nltk.collocations.BigramAssocMeasures()
	tokens = nltk.wordpunct_tokenize(text)   
	# print tokens 
	# tokens = nltk.wordpunct_tokenize(text.translate(None, string.punctuation))     
	bi_finder = BigramCollocationFinder.from_words(tokens)
	bi_finder.apply_word_filter(lambda w: w in STOP_WORDS_SET)
	bi_scored = bi_finder.score_ngrams(bigram_measures.student_t)
	score, val = 0, 0
	for key, scores in bi_scored:
		score += scores
	if len(tokens)!=0:
		val = score/len(tokens)
	# print val,  " Bigram Score\n"
	return val

def sym(sentence, s2):
	sentence = [i for i in sentence.split() if i not in STOP_WORDS_SET]
	for i in sentence:
		i = stemmer.stem(i)
	# print sentence
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
			and word.lower() not in STOP_WORDS_SET)
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
	# print "tree:",len(tree), "toks:", len(toks), "postoks:", len(postoks), "score:", score
	return float(score)/len(toks)
	#print corpus.document_topic_prob
	#print corpus.topic_word_prob
	# topic_topk = 20
	# document_topk = 
	#print_topic_word_distribution(corpus, number_of_topics, topic_topk, "./topic-word.txt")
	#print_document_topic_distribution(corpus, number_of_topics, document_topk, "./document-topic.txt")
	
if __name__ == "__main__":
	main(sys.argv)
