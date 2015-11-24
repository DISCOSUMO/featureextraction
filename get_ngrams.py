# python get_ngrams.py 3 ../dataconversion/reddit_data/samples/200randomthreads freqlists_reddit_200_threads.out

import sys
import os
#import nltk.data
reload(sys)
sys.setdefaultencoding('utf8')
import re
import math
import fileinput
from collections import defaultdict
import xml.etree.ElementTree as ET

rootdir = sys.argv[2]
outfile = sys.argv[3]
maxn = sys.argv[1]

def tokenize(t) :
    text = t.lower()
    text = re.sub("[^a-zA-Z0-9- ']", "", text)
#    print text
    words = text.split()
    return words

def printTopN(topn,dict,out) :
    i = 0
    for term in sorted(dict,key=dict.get,reverse=True) :
        i = i+1
        if (i > topn) :
            break
        out.write(term+"\t"+str(dict[term])+"\n")

terms = dict()
unigrams = dict()
bigrams = dict()
trigrams = dict()

def getAllTerms (text) :
    words = tokenize(text)
    i=0
    #print len(words)
    #print rwords
    for word in words :
        if word in terms.keys() :
            terms[word] += 1
            unigrams[word] += 1
        else :
            terms[word] = 1
            unigrams[word] = 1

        if maxn >= 2 :
            if i< len(words)-1 :
                bigram = words[i]+" "+words[i+1]
                if bigram in terms.keys() :
                    terms[bigram] += 1
                    bigrams[bigram] += 1
                else :
                    terms[bigram] = 1
                    bigrams[bigram] = 1

                if maxn >= 3 :
                    if i < len(words)-2 :
                        #print i, words[i], words[i+1], words[i+2]
                        trigram = words[i]+" "+words[i+1]+" "+words[i+2]
                        if trigram in terms.keys() :
                            terms[trigram] += 1
                            trigrams[trigram] += 1
                        else :
                            terms[trigram] = 1
                            trigrams[trigram] =1
        i += 1


fc=0
for filename in os.listdir(rootdir):
    if filename.endswith("xml") :
        fc += 1
        print fc, filename
        if "xml" in filename :
            tree = ET.parse(rootdir+"/"+filename)
            root = tree.getroot()
            for thread in root:
                title = thread.find('title').text
                getAllTerms(title)
                for posts in thread.findall('posts'):
                    firstpost = posts.findall('post')[0]
                    body = firstpost.find('body').text
                    getAllTerms(body)
                #    for post in posts.findall('post'):
                #        body = post.find('body').text
                #        if body is None :
                #            continue
                #        else :
                #            getAllTerms(body)
                            #print body




out = open(outfile,"w")

out.write ("ALL TERMS\n")
printTopN(100,terms,out)

out.write ("\nUNIGRAMS\n")
printTopN(100,unigrams,out)

out.write ("\nBIGRAMS\n")
printTopN(50,bigrams,out)

out.write ("\nTRIGRAMS\n")
printTopN(20,trigrams,out)
