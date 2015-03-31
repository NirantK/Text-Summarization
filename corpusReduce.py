from requests import get
import glob
import os
import io
import nltk
import random
import numpy

def median(lst):
	return numpy.median(numpy.array(lst))

sss_url = "http://swoogle.umbc.edu/SimService/GetSimilarity"

def sss(s1, s2, type='relation', corpus='webbase'):
	try:
		response = get(sss_url, params={'operation':'api','phrase1':s1,'phrase2':s2,'type':type,'corpus':corpus})
		return float(response.text.strip())
	except:
		# print 'Error in getting similarity for %s: %s' % ((s1,s2))
		return -1.0

def test(s1, s2):
	return random.random()

def driver():
	linedump = []
	datapath = ['./dataset/summaries/test']
	for document_path in datapath:
		for document_file in glob.glob(os.path.join(document_path, '*.txt')):
			mockingJay = []
			try:
				with io.open(document_file, "r+", encoding='utf-8') as f:
					document = f.read()
			except Exception as e:
				print "Error reading file:",e
			textlines = nltk.sent_tokenize(document)
			for ele in textlines:
				linedump.append([document_file, ele])
	print len(linedump)
	writeString = ""
	valList = []
	with open("sssvalue.txt", "w") as f:
		for i in xrange(len(linedump)):
			for j in xrange(i + 1, len(linedump)):
				val = test (linedump[i], linedump[j])
				valList.append(val)
				writeString+=str(i) +","+ str(j)+","+ str(val)
				writeString+="\n"
		f.write(writeString)

	medianVal = median(valList)
	stdVal = numpy.std(numpy.array(valList))
	toDelIndices = []
	with open("sssvalue.txt", "r+") as f:
		# do = f.read()
		lines  = f.readlines()
		for ele in lines:
			lst = ele.strip().split(',')
			if float(lst[2]) - medianVal > 1.732*stdVal:
				toDelIndices.append(float(lst[0]))

	remove = []
	for ele in toDelIndices:
		ele = str(ele)
		ele = ele.split('.')[0]
		remove.append(int(ele))
	remove = list(set(remove))
	print len(remove)
	for index in sorted(remove, reverse=True):
		# print 'removing', index, linedump[index][0].encode('utf-8')
		del linedump[index]
	for ele in linedump:
		# print 'Truncating file', ele[0]
		open(ele[0], 'w').close()
	for ele in linedump:
		f = io.open(ele[0], 'a+', encoding="utf-8")
		# print 'File opened for rewrite:', ele[0]
		f.write(ele[1])
		f.write(u"\n")

if __name__ == "__main__":
	driver()