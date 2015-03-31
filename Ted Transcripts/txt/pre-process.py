from os import listdir
import os
from os.path import isfile, join
mypath = os.getcwd()
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
SHORTEST_SENTENCE_LENGTH = 5
for filename in onlyfiles:
	f = open(filename,"r")
	x = f.readlines()
	j = []
	l = len(x)
	for i in xrange(l):
		x[i] = " ".join(x[i].strip().split()[6:])
		if len(x[i].split())< SHORTEST_SENTENCE_LENGTH:
			x[i-1] += " " + x[i]
			j.append(x[i])

	for ele in x:
		if ele in j:
			x.remove(ele)
	f.close()

	f2 = "".join(filename.split(".")[0])+".txt"
	# print f2
	f = open(f2, "w+")
	for line in x:
		f.write(line+"\n")